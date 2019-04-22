# -*- coding: utf-8 -*-
import boto3
import json
from botocore.vendored import requests
from requests_aws4auth import AWS4Auth

region = 'us-east-1'  # For example, us-west-1
service = 'es'
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)

host = 'search-myrestaurants-cvudxasaxuxwxfy7d2kcwrluf4.us-east-1.es.amazonaws.com'  # For example, search-mydomain-id.us-west-1.es.amazonaws.com
index = 'yelp-restaurants'
url = 'https://' + host + '/' + index + '/_search'

ses = boto3.client('ses', region)

email_from = 'xiaoxy0501@gmail.com'
email_to = 'Email'
email_cc = 'Email'
emaiL_subject = 'Restaurants Suggestions'
email_body = 'Body'


def lambda_handler(event, context):
    # Create SQS client
    sqs = boto3.client('sqs')

    queue_url = 'https://sqs.us-east-1.amazonaws.com/791032249995/my-restaurants-queue.fifo'

    # Receive message from SQS queue
    response = sqs.receive_message(
        QueueUrl=queue_url,
        AttributeNames=[
            'SentTimestamp'
        ],
        MaxNumberOfMessages=1,
        MessageAttributeNames=[
            'All'
        ],
        VisibilityTimeout=90,
        WaitTimeSeconds=0
    )

    message = response['Messages'][0]
    receipt_handle = message['ReceiptHandle']
    cuisine = message['MessageAttributes'].get('Cuisine').get('StringValue')
    email = message['MessageAttributes'].get('Email').get('StringValue')
    number = message['MessageAttributes'].get('Number').get('StringValue')
    diningdate = message['MessageAttributes'].get('DiningDate').get('StringValue')
    diningtime = message['MessageAttributes'].get('DiningTime').get('StringValue')

    # Delete received message from queue
    sqs.delete_message(
        QueueUrl=queue_url,
        ReceiptHandle=receipt_handle
    )
    # return('Received and deleted message: %s' % attributes)

    # Put the user query into the query DSL for more accurate search results.
    # Note that certain fields are boosted (^).
    query = {
        "size": 3,
        "query": {
            "match": {
                "Cuisine_Type": cuisine
            }
        }
    }

    # ES 6.x requires an explicit Content-Type header
    headers = {"Content-Type": "application/json"}

    # Make the signed HTTP request
    r = requests.get(url, auth=awsauth, headers=headers, data=json.dumps(query))

    # Create the response and add some extra content to support CORS
    response = {
        "statusCode": 200,
        "headers": {
            "Access-Control-Allow-Origin": '*'
        },
        "isBase64Encoded": False
    }

    # Add the search results to the response
    data = [doc for doc in json.loads(r.text)['hits']['hits']]
    Business_ID = []
    for doc in data:
        Business_ID.append(doc['_id'].split('=')[1])

    # response['body'] = Business_ID

    # SES message
    email_body = 'Hello! Here are my %s restaurant suggestions for %s people, for %s at %s:' % (
    cuisine, number, diningdate, diningtime)

    dynamodb = boto3.client('dynamodb')
    i = 1
    for id in Business_ID:
        db_data = dynamodb.get_item(TableName='yelp-restaurants', Key={'Business_ID': {'S': id}})
        email_body = email_body + ' ' + str(i) + '. ' + db_data['Item']['Name']['S'] + ', located at ' + \
                     db_data['Item']['Address']['S']
        i += 1
    email_body += '. Enjoy your meal!'
    response['body'] = email_body

    # email_to="cindyxiao0501@gmail.com"
    email_to = email
    ses.send_email(
        Source=email_from,
        Destination={
            'ToAddresses': [
                email_to,
            ],
            # 'CcAddresses': [
            #     email_cc,
            # ]
        },
        Message={
            'Subject': {
                'Data': emaiL_subject
            },
            'Body': {
                'Text': {
                    'Data': email_body
                }
            }
        }
    )
    return response

