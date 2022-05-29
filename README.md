# imagenet-viewer-encoder

The service convert images to feature vectors (encode images) which is required 
for further matching and clustering.

# Requirements

1. Ubuntu 20.04.3
2. Python 3.8.10
3. requirements.txt
4. python -m pip install git+https://github.com/yashenkoxciv/imagenet-viewer.git


# Expected environment variables

| Name                | Description                                                               |
|---------------------|:--------------------------------------------------------------------------|
| TRITON_HOST         | Triton's host                                                             |
| MODEL_NAME          | deployed model name for feature extraction (not classification)           |
| RABBITMQ_HOST       | RabbitMQ's host                                                           |
| INPUT_QUEUE         | RabbitMQ's queue with images to encode                                    |
| OUTPUT_QUEUE        | RabbitMQ's queue to push encoded image                                    |
| MONGODB_HOST        | MongoDB's connection string like this: mongodb://host:port/imagenetviewer |
| MILVUS_ALIAS        | Milvus's connection alias                                                 |
| MILVUS_HOST         | Milvus's server host                                                      |
| MILVUS_PORT         | Milvus's server port                                                      |

