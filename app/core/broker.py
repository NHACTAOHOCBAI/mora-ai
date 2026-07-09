import json
import threading
import pika
from loguru import logger
from app.core.config import settings

EXCHANGE = "mora.direct.exchange"
USER_QUESTION_QUEUE = "mora.queue.user-question"
ANSWER_VERIFIED_QUEUE = "mora.queue.answer-verified"

USER_QUESTION_ROUTING_KEY = "mora.route.user-question"
ANSWER_VERIFIED_ROUTING_KEY = "mora.route.answer-verified"

class EventBroker:
    def __init__(self):
        self.connection = None
        self.channel = None
        self.consumer_thread = None

    def _connect(self):
        credentials = pika.PlainCredentials(settings.rabbitmq_user, settings.rabbitmq_password)
        parameters = pika.ConnectionParameters(
            host=settings.rabbitmq_host,
            port=settings.rabbitmq_port,
            credentials=credentials,
            heartbeat=600
        )
        self.connection = pika.BlockingConnection(parameters)
        self.channel = self.connection.channel()

        # Declare exchange and queues to ensure they exist
        self.channel.exchange_declare(exchange=EXCHANGE, exchange_type='direct', durable=True)
        self.channel.queue_declare(queue=USER_QUESTION_QUEUE, durable=True)
        self.channel.queue_declare(queue=ANSWER_VERIFIED_QUEUE, durable=True)
        
        self.channel.queue_bind(queue=USER_QUESTION_QUEUE, exchange=EXCHANGE, routing_key=USER_QUESTION_ROUTING_KEY)
        self.channel.queue_bind(queue=ANSWER_VERIFIED_QUEUE, exchange=EXCHANGE, routing_key=ANSWER_VERIFIED_ROUTING_KEY)

    def publish_answer_verified(self, event_data: dict):
        try:
            # Create a separate connection for publishing to prevent thread sharing issues
            credentials = pika.PlainCredentials(settings.rabbitmq_user, settings.rabbitmq_password)
            parameters = pika.ConnectionParameters(
                host=settings.rabbitmq_host,
                port=settings.rabbitmq_port,
                credentials=credentials
            )
            conn = pika.BlockingConnection(parameters)
            ch = conn.channel()
            
            logger.info(f"Publishing AnswerVerifiedEvent: assistantMessageId={event_data.get('assistantMessageId')}")
            ch.basic_publish(
                exchange=EXCHANGE,
                routing_key=ANSWER_VERIFIED_ROUTING_KEY,
                body=json.dumps(event_data),
                properties=pika.BasicProperties(
                    delivery_mode=2  # Make message persistent
                )
            )
            conn.close()
        except Exception as e:
            logger.error(f"Failed to publish AnswerVerifiedEvent: {e}", exc_info=True)

    def start_consumer(self, callback_func):
        def run():
            while True:
                try:
                    logger.info("Connecting to RabbitMQ...")
                    self._connect()
                    
                    def on_message(ch, method, properties, body):
                        try:
                            event_data = json.loads(body.decode('utf-8'))
                            logger.info(f"Received UserQuestionEvent: userMessageId={event_data.get('userMessageId')}")
                            
                            # Execute the agent orchestrator
                            callback_func(event_data)
                            
                            ch.basic_ack(delivery_tag=method.delivery_tag)
                        except Exception as ex:
                            logger.error(f"Error handling message: {ex}", exc_info=True)
                            # Requeue on error so we don't lose the request
                            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

                    self.channel.basic_qos(prefetch_count=1)
                    self.channel.basic_consume(queue=USER_QUESTION_QUEUE, on_message_callback=on_message)
                    logger.info("RabbitMQ Consumer started successfully. Listening for user questions...")
                    self.channel.start_consuming()
                except pika.exceptions.AMQPConnectionError as e:
                    logger.warning(f"RabbitMQ connection lost, retrying in 5 seconds... Error: {e}")
                    import time
                    time.sleep(5)
                except Exception as e:
                    logger.error(f"Unexpected error in consumer thread: {e}", exc_info=True)
                    import time
                    time.sleep(5)

        self.consumer_thread = threading.Thread(target=run, daemon=True)
        self.consumer_thread.start()

event_broker = EventBroker()
