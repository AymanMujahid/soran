from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from typing import Generator, Optional
import logging
import time
from config import settings, DATABASE_CONFIG
from models import Base

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self._initialize_database()
    
    def _initialize_database(self):
        try:
            if settings.DATABASE_URL.startswith("sqlite"):
                self.engine = create_engine(
                    settings.DATABASE_URL,
                    poolclass=StaticPool,
                    connect_args={"check_same_thread": False},
                    echo=settings.DEBUG
                )
            else:
                self.engine = create_engine(
                    settings.DATABASE_URL,
                    **DATABASE_CONFIG,
                    echo=settings.DEBUG
                )
            
            self._setup_event_listeners()
            
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            logger.info("Database connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _setup_event_listeners(self):
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            if settings.DATABASE_URL.startswith("sqlite"):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA cache_size=1000")
                cursor.execute("PRAGMA temp_store=memory")
                cursor.close()
        
        @event.listens_for(self.engine, "checkout")
        def receive_checkout(dbapi_connection, connection_record, connection_proxy):
            connection_record.info['checkout_time'] = time.time()
        
        @event.listens_for(self.engine, "checkin")
        def receive_checkin(dbapi_connection, connection_record):
            if 'checkout_time' in connection_record.info:
                checkout_time = connection_record.info['checkout_time']
                duration = time.time() - checkout_time
                if duration > 30:
                    logger.warning(f"Long-running database connection: {duration:.2f}s")
    
    def create_tables(self):
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    def drop_tables(self):
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop database tables: {e}")
            raise
    
    def get_session(self) -> Session:
        if not self.SessionLocal:
            raise RuntimeError("Database not initialized")
        return self.SessionLocal()
    
    def health_check(self) -> bool:
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def close(self):
        if self.engine:
            self.engine.dispose()
            logger.info("Database connections closed")

db_manager = DatabaseManager()

def get_db() -> Generator[Session, None, None]:
    session = db_manager.get_session()
    try:
        yield session
    except SQLAlchemyError as e:
        logger.error(f"Database session error: {e}")
        session.rollback()
        raise
    except Exception as e:
        logger.error(f"Unexpected error in database session: {e}")
        session.rollback()
        raise
    finally:
        session.close()

@contextmanager
def get_db_session():
    session = db_manager.get_session()
    try:
        yield session
        session.commit()
    except SQLAlchemyError as e:
        logger.error(f"Database transaction error: {e}")
        session.rollback()
        raise
    except Exception as e:
        logger.error(f"Unexpected error in database transaction: {e}")
        session.rollback()
        raise
    finally:
        session.close()

class DatabaseRepository:
    def __init__(self, session: Session):
        self.session = session
    
    def create(self, model_class, **kwargs):
        try:
            instance = model_class(**kwargs)
            self.session.add(instance)
            self.session.commit()
            self.session.refresh(instance)
            return instance
        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Error creating {model_class.__name__}: {e}")
            raise
    
    def get_by_id(self, model_class, id_value):
        try:
            return self.session.query(model_class).filter(model_class.id == id_value).first()
        except SQLAlchemyError as e:
            logger.error(f"Error getting {model_class.__name__} by id {id_value}: {e}")
            raise
    
    def get_all(self, model_class, limit: Optional[int] = None, offset: int = 0):
        try:
            query = self.session.query(model_class)
            if limit:
                query = query.limit(limit).offset(offset)
            return query.all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting all {model_class.__name__}: {e}")
            raise
    
    def update(self, instance, **kwargs):
        try:
            for key, value in kwargs.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)
            self.session.commit()
            self.session.refresh(instance)
            return instance
        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Error updating {instance.__class__.__name__}: {e}")
            raise
    
    def delete(self, instance):
        try:
            self.session.delete(instance)
            self.session.commit()
            return True
        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Error deleting {instance.__class__.__name__}: {e}")
            raise
    
    def bulk_create(self, model_class, data_list):
        try:
            instances = [model_class(**data) for data in data_list]
            self.session.bulk_save_objects(instances)
            self.session.commit()
            return len(instances)
        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Error bulk creating {model_class.__name__}: {e}")
            raise
    
    def execute_raw_sql(self, sql: str, params: dict = None):
        try:
            result = self.session.execute(sql, params or {})
            self.session.commit()
            return result
        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Error executing raw SQL: {e}")
            raise

class ConnectionMonitor:
    def __init__(self):
        self.active_connections = 0
        self.total_connections = 0
        self.failed_connections = 0
    
    def on_connect(self):
        self.active_connections += 1
        self.total_connections += 1
    
    def on_disconnect(self):
        self.active_connections -= 1
    
    def on_error(self):
        self.failed_connections += 1
    
    def get_stats(self):
        return {
            "active_connections": self.active_connections,
            "total_connections": self.total_connections,
            "failed_connections": self.failed_connections
        }

connection_monitor = ConnectionMonitor()

@event.listens_for(db_manager.engine, "connect")
def on_connect(dbapi_connection, connection_record):
    connection_monitor.on_connect()

@event.listens_for(db_manager.engine, "close")
def on_disconnect(dbapi_connection, connection_record):
    connection_monitor.on_disconnect()

@event.listens_for(db_manager.engine, "handle_error")
def on_error(exception_context):
    connection_monitor.on_error()

def init_database():
    try:
        db_manager.create_tables()
        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False

def reset_database():
    try:
        db_manager.drop_tables()
        db_manager.create_tables()
        logger.info("Database reset successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to reset database: {e}")
        return False

async def database_health_check():
    return db_manager.health_check()

def get_connection_stats():
    engine_pool = db_manager.engine.pool
    return {
        "pool_size": engine_pool.size(),
        "checked_in": engine_pool.checkedin(),
        "overflow": engine_pool.overflow(),
        "checked_out": engine_pool.checkedout(),
        "monitor_stats": connection_monitor.get_stats()
    }