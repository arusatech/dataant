import boto3
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from typing import Optional, List, Dict, Any
from jsonpath_nz import log, jprint
from boto3.dynamodb.conditions import Key, Attr

class DatabaseEngine:
    """Database engine factory for multiple database types."""
    
    def __init__(self, db_type: str, **kwargs):
        self.db_type = db_type.lower()
        self.connection = self._create_connection(kwargs)
        
    def _create_connection(self, params: dict) -> Any:
        """Create appropriate database connection based on type."""
        try:
            if self.db_type == 'dynamodb':
                log.info(params)
                # region = params.get('region', 'us-west-2')
                environment = params.get('environment', 'dev')
                # db_name = params.get('db_name', 'loan_rate')
                # Create both resource and client for different operations
                session = boto3.Session(profile_name=environment)
                self.dynamodb_resource = session.resource('dynamodb')
                self.dynamodb_client = session.client('dynamodb')
                return self.dynamodb_resource
                
            else:
                # For SQL databases, create SQLAlchemy engine
                if self.db_type == 'sqlite':
                    path = params.get('path', 'database.db')
                    url = f'sqlite:///{path}'
                elif self.db_type == 'postgresql':
                    url = (f"postgresql://{params['user']}:{params['password']}@"
                          f"{params['host']}:{params.get('port', 5432)}/{params['database']}")
                elif self.db_type == 'mysql':
                    url = (f"mysql://{params['user']}:{params['password']}@"
                          f"{params['host']}:{params.get('port', 3306)}/{params['database']}")
                else:
                    raise ValueError(f"Unsupported database type: {self.db_type}")
                
                self.engine = create_engine(url, echo=params.get('echo', False))
                self.Session = sessionmaker(bind=self.engine)
                self.Base = declarative_base()
                return self.engine
                
        except Exception as e:
            log.error(f"Failed to create connection for {self.db_type}: {str(e)}")
            raise

    def list_items(self, table_name: str, key_condition_expression=None, 
                   filter_expression=None, limit: Optional[int] = None) -> List[Dict]:
        """
        Query items from a DynamoDB table.
        
        Args:
            table_name: Name of the DynamoDB table
            key_condition_expression: Required. The condition for the partition key (and sort key if used)
            filter_expression: Optional. Additional filters after the query
            limit: Optional. Maximum number of items to return
            
        Returns:
            List of items from the table
        """
        if self.db_type != 'dynamodb':
            raise ValueError("list_items is only supported for DynamoDB")
            
        try:
            table = self.dynamodb_resource.Table(table_name)
            query_params = {
                'KeyConditionExpression': key_condition_expression
            }
            
            if limit:
                query_params['Limit'] = limit
            if filter_expression:
                query_params['FilterExpression'] = filter_expression
                
            response = table.query(**query_params)
            items = response.get('Items', [])
            
            # Handle pagination
            while 'LastEvaluatedKey' in response:
                query_params['ExclusiveStartKey'] = response['LastEvaluatedKey']
                response = table.query(**query_params)
                items.extend(response.get('Items', []))
                
                if limit and len(items) >= limit:
                    items = items[:limit]
                    break
                    
            log.info(f"Retrieved {len(items)} items from table {table_name}")
            return items
            
        except Exception as e:
            log.error(f"Error querying items from table {table_name}: {str(e)}")
            raise

    def list_tables(self) -> List[str]:
        """List all DynamoDB tables."""
        if self.db_type != 'dynamodb':
            raise ValueError("list_tables is only supported for DynamoDB")
            
        try:
            response = self.dynamodb_client.list_tables()
            return response.get('TableNames', [])
        except Exception as e:
            log.error(f"Error listing tables: {str(e)}")
            raise

    def get_table(self, table_name: str):
        """Get DynamoDB table resource."""
        if self.db_type != 'dynamodb':
            raise ValueError("get_table is only supported for DynamoDB")
        return self.dynamodb_resource.Table(table_name)

    def dispose(self):
        """Clean up resources."""
        if hasattr(self, 'engine'):
            self.engine.dispose()


# Usage example
if __name__ == "__main__":
    # Initialize DynamoDB connection
    dynamo_db = DatabaseEngine('dynamodb', environment='dev')
    
    try:
        key_condition = None
        filter_expr = None
        # Example query with partition key
        key_condition = Key('loan_number').eq('6666333111')
        
        # # Optional filter
        # filter_expr = Attr('rate').gt(5.0)
        
        # Query items
        items = dynamo_db.list_items(
            table_name='loan_rate',
            key_condition_expression=key_condition,
            filter_expression=filter_expr,
            limit=10
        )
        print(f"Query results: {items}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        dynamo_db.dispose()

