import os
import pika
import logging
import numpy as np
from bson import ObjectId
from environs import Env
from mongoengine import connect, disconnect
from imagenetviewer.image import Image, ImageStatus
from inference import TritonImageNetFeatureExtractorModel
from imagenetviewer.vector import ImagesFeatures
from pymilvus import connections


def on_request(ch, method, props, body):
    image_object_id = ObjectId(body.decode())
    image = Image.objects.get(pk=image_object_id)

    z = triton.inference(triton.preprocessing(image.get_pil_image()))
    z = z / np.linalg.norm(z, axis=1, keepdims=True)

    mr = imgf_collection.insert_vectors(z)
    image.vector_id = mr[0]
    image.status = ImageStatus.PENDING_MATCHING

    image.save()
    logger.info(f'{image_object_id} encoded')

    ch.basic_publish(
        exchange='',
        routing_key=env('OUTPUT_QUEUE'),
        body=str(image_object_id)
    )

    ch.basic_ack(delivery_tag=method.delivery_tag)



if __name__ == '__main__':
    logger = logging.getLogger('encoder')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(asctime)s %(name)s %(levelname)s] %(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    env = Env()
    env.read_env()

    triton = TritonImageNetFeatureExtractorModel(env('TRITON_HOST'), env('MODEL_NAME'))

    connect(host=env('MONGODB_HOST'), uuidRepresentation='standard')

    connections.connect(env('MILVUS_ALIAS'), host=env('MILVUS_HOST'), port=env('MILVUS_PORT'))
    imgf_collection = ImagesFeatures()
    imgf_collection.collection.load()

    con_par = pika.ConnectionParameters(
        heartbeat=600,
        blocked_connection_timeout=300,
        host=env('RABBITMQ_HOST')
    )
    connection = pika.BlockingConnection(con_par)
    channel = connection.channel()

    channel.queue_declare(queue=env('INPUT_QUEUE'), durable=True)
    channel.queue_declare(queue=env('OUTPUT_QUEUE'), durable=True)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=env('INPUT_QUEUE'), on_message_callback=on_request)

    logger.info('[+] awaiting image to beat features out of it')
    channel.start_consuming()

    connections.disconnect(env('MILVUS_ALIAS'))



