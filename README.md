# CloudComputing HW 2

**1. Use the Yelp API to collect 5,000+ random restaurants from Manhattan.**

- use python to scrape the data from Yelp API

```python
# -*- coding: utf-8 -*-
"""
Yelp Fusion API code sample.
This program demonstrates the capability of the Yelp Fusion API
by using the Search API to query for businesses by a search term and location,
and the Business API to query additional information about the top result
from the search query.
Please refer to http://www.yelp.com/developers/v3/documentation for the API
documentation.
This program requires the Python requests library, which you can install via:
`pip install -r requirements.txt`.
Sample usage of the program:
`python sample.py --term="bars" --location="San Francisco, CA"`
"""
from __future__ import print_function

import argparse
import json
import pprint
import requests
import sys
import urllib
import csv


# This client code can run on Python 2.x or 3.x.  Your imports can be
# simpler if you only need one of those.
try:
    # For Python 3.0 and later
    from urllib.error import HTTPError
    from urllib.parse import quote
    from urllib.parse import urlencode
except ImportError:
    # Fall back to Python 2's urllib2 and urllib
    from urllib import HTTPError
    from urllib import quote
    from urllib import urlencode


# Yelp Fusion no longer uses OAuth as of December 7, 2017.
# You no longer need to provide Client ID to fetch Data
# It now uses private keys to authenticate requests (API Key)
# You can find it on
# https://www.yelp.com/developers/v3/manage_app
API_KEY= '7YLMixP_6HJAQwa-I0eYjiNQTKO1Lv2WBKmJxDzAhP-jQKqeFaiWXBSyOcjaEZyuq-VkLX9q5Nu-_Bw11M1f59DoiOwHWyqrZhKKy2zRC0Bu06k9edCXv2L04-azXHYx'



# API constants, you shouldn't have to change these.
API_HOST = 'https://api.yelp.com'
SEARCH_PATH = '/v3/businesses/search'
BUSINESS_PATH = '/v3/businesses/'  # Business ID will come after slash.


# Defaults for our simple example.
DEFAULT_TERM = 'Chinese'
DEFAULT_LOCATION = 'Manhattan, NY'
SEARCH_LIMIT = 50


def request(host, path, api_key, url_params=None):
    """Given your API_KEY, send a GET request to the API.
    Args:
        host (str): The domain host of the API.
        path (str): The path of the API after the domain.
        API_KEY (str): Your API Key.
        url_params (dict): An optional set of query parameters in the request.
    Returns:
        dict: The JSON response from the request.
    Raises:
        HTTPError: An error occurs from the HTTP request.
    """
    url_params = url_params or {}
    url = '{0}{1}'.format(host, quote(path.encode('utf8')))
    headers = {
        'Authorization': 'Bearer %s' % api_key,
    }

    print(u'Querying {0} ...'.format(url))

    response = requests.request('GET', url, headers=headers, params=url_params)

    return response.json()


def search(api_key, term, location, offset):
    """Query the Search API by a search term and location.
    Args:
        term (str): The search term passed to the API.
        location (str): The search location passed to the API.
    Returns:
        dict: The JSON response from the request.
    """

    url_params = {
        'term': term.replace(' ', '+'),
        'location': location.replace(' ', '+'),
        'limit': SEARCH_LIMIT,
        'offset': offset
    }
    return request(API_HOST, SEARCH_PATH, api_key, url_params=url_params)


def get_business(api_key, business_id):
    """Query the Business API by a business ID.
    Args:
        business_id (str): The ID of the business to query.
    Returns:
        dict: The JSON response from the request.
    """
    business_path = BUSINESS_PATH + business_id

    return request(API_HOST, business_path, api_key)


def query_api(term, location):
    """Queries the API by the input values from the user.
    Args:
        term (str): The search term to query.
        location (str): The location of the business to query.
    """
    for cnt in range(20):
        data=[]
        response = search(API_KEY, term, location,cnt*50)

        businesses = response.get('businesses')

        if not businesses:
            print(u'No businesses for {0} in {1} found.'.format(term, location))
            return

        for i in range(len(businesses)):
            business_id = businesses[i]['id']
            print(u'{0} businesses found, querying business info ' \
            'for the top result "{1}" ...'.format(len(businesses), business_id))
            response = get_business(API_KEY, business_id)

            print(u'Result for business "{0}" found:'.format(business_id))
            # pprint.pprint(response, indent=2)
            dict_after={'Business_ID':response['id'],'Cuisine_Type':term,'Name':response['alias'],'Address':', '.join(response['location']['display_address']),\
                        'Coordinates':(response['coordinates']['latitude'],response['coordinates']['longitude']),'Number_of_Reviews':response['review_count'],'Rating':response['rating'],'ZipCode':response['location']['zip_code']}
            data.append(dict_after)
        pprint.pprint(data, indent=2)
        write_data(data)

def write_data(data):
    yelp_data = open('yelp.csv', 'a+')
    csvwriter = csv.writer(yelp_data)
    for item in data:
        csvwriter.writerow(item.values())
    yelp_data.close()


def main(term):
    parser = argparse.ArgumentParser()

    parser.add_argument('-q', '--term', dest='term', default=term,
                        type=str, help='Search term (default: %(default)s)')
    parser.add_argument('-l', '--location', dest='location',
                        default=DEFAULT_LOCATION, type=str,
                        help='Search location (default: %(default)s)')

    input_values = parser.parse_args()

    try:
        query_api(input_values.term, input_values.location)
    except HTTPError as error:
        sys.exit(
            'Encountered HTTP error {0} on {1}:\n {2}\nAbort program.'.format(
                error.code,
                error.url,
                error.read(),
            )
        )


if __name__ == '__main__':
    yelp_data = open('test.csv', 'w')
    csvwriter = csv.writer(yelp_data)
    header = ['Business_ID','Cuisine_Type','Name','Address','Coordinates','Number_of_Reviews','Rating']
    csvwriter.writerow(header)
    yelp_data.close()
    main('Chinese')
    main('American')
    main('Mexican')
    main('Italian')
    main('Korean')
    main('France')
    # main('Indian') 
```

2. **Store data into DynamoDB **

- use python to transfer data in csv to DynamoDB

```python
# -*- coding: utf-8 -*-
import boto3
import csv
import datetime
dynamodb=boto3.resource('dynamodb','us-east-1')

def batch_write(table_name,rows):
    table=dynamodb.Table(table_name)

    with table.batch_writer() as batch:
        for row in rows:
            batch.put_item(Item=row)
    return True

def read_csv(csv_file,list):
    rows=csv.DictReader(open(csv_file))
    for row in rows:
        row['insertedAtTimestamp']=datetime.datetime.now().isoformat()
        print(row)
        list.append(row)
if __name__ == '__main__':
    table_name='yelp-restaurants'
    file_name='yelp.csv'
    items=[]
    read_csv(file_name, items)
    status=batch_write(table_name, items)
    if(status):
        print('Data is saved')
    else:
        print('Error while inserting data')

```

3.**Create an ElasticSearch instance using the AWS ElasticSearch Service.**

- create lambda function to  index new and existing Amazon DynamoDB content with Amazon Elasticsearch Service

![image-20190421203223249](/Users/cong/Library/Application Support/typora-user-images/image-20190421203223249.png)

```python
import base64
import datetime
import json
import logging
import os
import time
import traceback
import urllib
import urlparse

from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import get_credentials
from botocore.endpoint import BotocoreHTTPSession
from botocore.session import Session
from boto3.dynamodb.types import TypeDeserializer


# The following parameters are required to configure the ES cluster
ES_ENDPOINT = 'search-myrestaurants-cvudxasaxuxwxfy7d2kcwrluf4.us-east-1.es.amazonaws.com'

# The following parameters can be optionally customized
DOC_TABLE_FORMAT = '{}'         # Python formatter to generate index name from the DynamoDB table name
DOC_TYPE_FORMAT = '{}_type'     # Python formatter to generate type name from the DynamoDB table name, default is to add '_type' suffix
ES_REGION = None                # If not set, use the runtime lambda region
ES_MAX_RETRIES = 3              # Max number of retries for exponential backoff
DEBUG = True                    # Set verbose debugging information

print "Streaming to ElasticSearch"
logger = logging.getLogger()
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)


# Subclass of boto's TypeDeserializer for DynamoDB to adjust for DynamoDB Stream format.
class StreamTypeDeserializer(TypeDeserializer):
   def _deserialize_n(self, value):
       return float(value)

   def _deserialize_b(self, value):
       return value  # Already in Base64


class ES_Exception(Exception):
   '''Exception capturing status_code from Client Request'''
   status_code = 0
   payload = ''

   def __init__(self, status_code, payload):
       self.status_code = status_code
       self.payload = payload
       Exception.__init__(self, 'ES_Exception: status_code={}, payload={}'.format(status_code, payload))


# Low-level POST data to Amazon Elasticsearch Service generating a Sigv4 signed request
def post_data_to_es(payload, region, creds, host, path, method='POST', proto='https://'):
   '''Post data to ES endpoint with SigV4 signed http headers'''
   req = AWSRequest(method=method, url=proto + host + urllib.quote(path), data=payload, headers={'Host': host, 'Content-Type' : 'application/json'})
   SigV4Auth(creds, 'es', region).add_auth(req)
   http_session = BotocoreHTTPSession()
   res = http_session.send(req.prepare())
   if res.status_code >= 200 and res.status_code <= 299:
       return res._content
   else:
       raise ES_Exception(res.status_code, res._content)


# High-level POST data to Amazon Elasticsearch Service with exponential backoff
# according to suggested algorithm: http://docs.aws.amazon.com/general/latest/gr/api-retries.html
def post_to_es(payload):
   '''Post data to ES cluster with exponential backoff'''

   # Get aws_region and credentials to post signed URL to ES
   es_region = ES_REGION or os.environ['AWS_REGION']
   session = Session({'region': es_region})
   creds = get_credentials(session)
   es_url = urlparse.urlparse(ES_ENDPOINT)
   es_endpoint = es_url.netloc or es_url.path  # Extract the domain name in ES_ENDPOINT

   # Post data with exponential backoff
   retries = 0
   while retries < ES_MAX_RETRIES:
       if retries > 0:
           seconds = (2 ** retries) * .1
           time.sleep(seconds)

       try:
           es_ret_str = post_data_to_es(payload, es_region, creds, es_endpoint, '/_bulk')
           es_ret = json.loads(es_ret_str)

           if es_ret['errors']:
               logger.error('ES post unsuccessful, errors present, took=%sms', es_ret['took'])
               # Filter errors
               es_errors = [item for item in es_ret['items'] if item.get('index').get('error')]
               logger.error('List of items with errors: %s', json.dumps(es_errors))
           else:
               logger.info('ES post successful, took=%sms', es_ret['took'])
           break  # Sending to ES was ok, break retry loop
       except ES_Exception as e:
           if (e.status_code >= 500) and (e.status_code <= 599):
               retries += 1  # Candidate for retry
           else:
               raise  # Stop retrying, re-raise exception


# Extracts the DynamoDB table from an ARN
# ex: arn:aws:dynamodb:eu-west-1:123456789012:table/table-name/stream/2015-11-13T09:23:17.104 should return 'table-name'
def get_table_name_from_arn(arn):
   return arn.split(':')[5].split('/')[1]


# Compute a compound doc index from the key(s) of the object in lexicographic order: "k1=key_val1|k2=key_val2"
def compute_doc_index(keys_raw, deserializer):
   index = []
   for key in sorted(keys_raw):
       index.append('{}={}'.format(key, deserializer.deserialize(keys_raw[key])))
   return '|'.join(index)


def _lambda_handler(event, context):
   records = event['Records']
   now = datetime.datetime.utcnow()

   ddb_deserializer = StreamTypeDeserializer()
   es_actions = []  # Items to be added/updated/removed from ES - for bulk API
   cnt_insert = cnt_modify = cnt_remove = 0
   for record in records:
       # Handle both native DynamoDB Streams or Streams data from Kinesis (for manual replay)
       if record.get('eventSource') == 'aws:dynamodb':
           ddb = record['dynamodb']
           ddb_table_name = get_table_name_from_arn(record['eventSourceARN'])
           doc_seq = ddb['SequenceNumber']
       elif record.get('eventSource') == 'aws:kinesis':
           ddb = json.loads(base64.b64decode(record['kinesis']['data']))
           ddb_table_name = ddb['SourceTable']
           doc_seq = record['kinesis']['sequenceNumber']
       else:
           logger.error('Ignoring non-DynamoDB event sources: %s', record.get('eventSource'))
           continue

       # Compute DynamoDB table, type and index for item
       doc_table = DOC_TABLE_FORMAT.format(ddb_table_name.lower())  # Use formatter
       doc_type = DOC_TYPE_FORMAT.format(ddb_table_name.lower())    # Use formatter
       doc_index = compute_doc_index(ddb['Keys'], ddb_deserializer)

       # Dispatch according to event TYPE
       event_name = record['eventName'].upper()  # INSERT, MODIFY, REMOVE

       # Treat events from a Kinesis stream as INSERTs
       if event_name == 'AWS:KINESIS:RECORD':
           event_name = 'INSERT'

       # Update counters
       if event_name == 'INSERT':
           cnt_insert += 1
       elif event_name == 'MODIFY':
           cnt_modify += 1
       elif event_name == 'REMOVE':
           cnt_remove += 1
       else:
           logger.warning('Unsupported event_name: %s', event_name)

       # If DynamoDB INSERT or MODIFY, send 'index' to ES
       if (event_name == 'INSERT') or (event_name == 'MODIFY'):
           if 'NewImage' not in ddb:
               logger.warning('Cannot process stream if it does not contain NewImage')
               continue
           # Deserialize DynamoDB type to Python types
           doc_fields = ddb_deserializer.deserialize({'M': ddb['NewImage']})
           # Add metadata
           doc_fields['@timestamp'] = now.isoformat()
           doc_fields['@SequenceNumber'] = doc_seq

           # Generate JSON payload
           doc_json = json.dumps(doc_fields)

           # Generate ES payload for item
           action = {'index': {'_index': doc_table, '_type': doc_type, '_id': doc_index}}
           es_actions.append(json.dumps(action))  # Action line with 'index' directive
           es_actions.append(doc_json)            # Payload line

       # If DynamoDB REMOVE, send 'delete' to ES
       elif event_name == 'REMOVE':
           action = {'delete': {'_index': doc_table, '_type': doc_type, '_id': doc_index}}
           es_actions.append(json.dumps(action))

   # Prepare bulk payload
   es_actions.append('')  # Add one empty line to force final \n
   es_payload = '\n'.join(es_actions)

   post_to_es(es_payload)  # Post to ES with exponential backoff


# Global lambda handler - catches all exceptions to avoid dead letter in the DynamoDB Stream
def lambda_handler(event, context):
   try:
       return _lambda_handler(event, context)
   except Exception:
       logger.error(traceback.format_exc())
```

- use post_to_es_from_dynamodb.py to add pre-existing content to Elasticsearch Service from DynamoDB.

```python
# -*- coding: utf-8 -*-
import json
import boto3
import boto3.dynamodb.types
import logging
import argparse
from boto3 import Session

logging.basicConfig()

client = boto3.client('lambda', region_name='us-east-1')
reports = []
object_amount = 0
partSize = 0


def main():
    parser = argparse.ArgumentParser(description='Set-up importing to dynamodb')
    parser.add_argument('--rn', metavar='R', help='AWS region')
    parser.add_argument('--tn', metavar='T', help='table name')
    parser.add_argument('--ak', metavar='AK', help='aws access key')
    parser.add_argument('--sk', metavar='AS', help='aws secret key')
    parser.add_argument('--esarn', metavar='ESARN', help='event source ARN')
    parser.add_argument('--lf', metavar='LF', help='lambda function that posts data to es')

    scan_limit = 300
    args = parser.parse_args()

    if (args.rn is None):
        print('Specify region parameter (-rn)')
        return

    client = boto3.client('lambda', region_name=args.rn)
    import_dynamodb_items_to_es(args.tn, args.sk, args.ak, args.rn, args.esarn, args.lf, scan_limit)


def import_dynamodb_items_to_es(table_name, aws_secret, aws_access, aws_region, event_source_arn, lambda_f, scan_limit):
    global reports
    global partSize
    global object_amount

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    session = Session(aws_access_key_id=aws_access, aws_secret_access_key=aws_secret, region_name=aws_region)
    dynamodb = session.resource('dynamodb')
    logger.info('dynamodb: %s', dynamodb)
    ddb_table_name = table_name
    table = dynamodb.Table(ddb_table_name)
    logger.info('table: %s', table)
    ddb_keys_name = [a['AttributeName'] for a in table.attribute_definitions]
    logger.info('ddb_keys_name: %s', ddb_keys_name)
    response = None

    while True:
        if not response:
            response = table.scan(Limit=scan_limit)
        else:
            response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'], Limit=scan_limit)
        for i in response["Items"]:
            ddb_keys = {k: i[k] for k in i if k in ddb_keys_name}
            ddb_data = boto3.dynamodb.types.TypeSerializer().serialize(i)["M"]
            ddb_keys = boto3.dynamodb.types.TypeSerializer().serialize(ddb_keys)["M"]
            record = {
                "dynamodb": {"SequenceNumber": "0000", "Keys": ddb_keys, "NewImage": ddb_data},
                "awsRegion": aws_region,
                "eventName": "INSERT",
                "eventSourceARN": event_source_arn,
                "eventSource": "aws:dynamodb"
            }
            partSize += 1
            object_amount += 1
            logger.info(object_amount)
            reports.append(record)

            if partSize >= 100:
                send_to_eslambda(reports, lambda_f)

        if 'LastEvaluatedKey' not in response:
            break

    if partSize > 0:
        send_to_eslambda(reports, lambda_f)


def send_to_eslambda(items, lambda_f):
    global reports
    global partSize
    records_data = {
        "Records": items
    }
    records = json.dumps(records_data)
    lambda_response = client.invoke(
        FunctionName=lambda_f,
        Payload=records
    )
    reports = []
    partSize = 0
    print(lambda_response)


if __name__ == "__main__":
    main()
# python post_to_es_from_dynamodb.py --tn "yelp-restaurants" --ak "AKIA3QLJSH2FQ2QKICUJ" --sk "rv5K8uw/7BRleW4k903XlpywgU72PzckX5H8YZnf" --esarn "arn:aws:dynamodb:us-east-1:791032249995:table/yelp-restaurants/stream/2019-04-19T17:55:31.170" --lf "arn:aws:lambda:us-east-1:791032249995:function:dynamodb-steam-to-es"

```

**7. Build a suggestions module, that is decoupled from the Lex chatbot.**

- In LF1, during the fulfillment step, push the information collected from the user (location, cuisine, etc.) to an SQS queue. 

![image-20190421204752048](/Users/cong/Library/Application Support/typora-user-images/image-20190421204752048.png)

```python
def sendSQSmessage(location, cuisine, number, dining_date, dining_time, email):
    # Create SQS client
    sqs = boto3.client('sqs')

    queue_url = 'https://sqs.us-east-1.amazonaws.com/791032249995/my-restaurants-queue.fifo'
    try:
        # Send message to SQS queue
        response = sqs.send_message(
            QueueUrl=queue_url,
            # DelaySeconds=123,
            MessageAttributes={
                'Location': {
                    'DataType': 'String',
                    'StringValue': location
                },
                'Cuisine': {
                    'DataType': 'String',
                    'StringValue': cuisine
                },
                'DiningDate': {
                    'DataType': 'String',
                    'StringValue': dining_date
                },
                'DiningTime': {
                    'DataType': 'String',
                    'StringValue': dining_time
                },
                'Email': {
                    'DataType': 'String',
                    'StringValue': email
                },
                'Number': {
                    'DataType': 'Number',
                    'StringValue': str(number)
                }
            },
            MessageBody=(
                 'Information about restaurant suggestion.'
            ),
            MessageDeduplicationId='string',
            MessageGroupId='string'
        )
    except IOError:print ("Error!")
    else:print ("Succeed!")

    
    print (response['MessageId'])

```

- Create a new Lambda function (LF2) that acts as a queue worker. Whenever it is invoked it 1. pulls a message from the SQS queue, 2. gets a random restaurant recommendation for the cuisine collected through conversation from ElasticSearch and DynamoDB, 3. formats them and 4. sends them over text message to the phone number included in the SQS message :

![image-20190421204832999](/Users/cong/Library/Application Support/typora-user-images/image-20190421204832999.png)

```python
import boto3
import json
from botocore.vendored import requests
from requests_aws4auth import AWS4Auth


region = 'us-east-1' # For example, us-west-1
service = 'es'
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)

host = 'search-myrestaurants-cvudxasaxuxwxfy7d2kcwrluf4.us-east-1.es.amazonaws.com' # For example, search-mydomain-id.us-west-1.es.amazonaws.com
index = 'yelp-restaurants'
url = 'https://' + host + '/' + index + '/_search'

ses = boto3.client('ses',region)

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
    cuisine=message['MessageAttributes'].get('Cuisine').get('StringValue')
    email=message['MessageAttributes'].get('Email').get('StringValue')
    number=message['MessageAttributes'].get('Number').get('StringValue')
    diningdate=message['MessageAttributes'].get('DiningDate').get('StringValue')
    diningtime=message['MessageAttributes'].get('DiningTime').get('StringValue')


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
    headers = { "Content-Type": "application/json" }

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
    Business_ID=[]
    for doc in data:
        Business_ID.append(doc['_id'].split('=')[1])
        
    # response['body'] = Business_ID
    
    #SES message
    email_body='Hello! Here are my %s restaurant suggestions for %s people, for %s at %s:' % (cuisine, number, diningdate, diningtime)
    
    dynamodb = boto3.client('dynamodb')
    i=1
    for id in Business_ID:
        db_data = dynamodb.get_item(TableName='yelp-restaurants', Key={'Business_ID':{'S':id}})
        email_body=email_body+' '+str(i)+'. '+db_data['Item']['Name']['S']+', located at '+db_data['Item']['Address']['S']
        i+=1
    email_body+='. Enjoy your meal!'
    response['body']=email_body

    # email_to="cindyxiao0501@gmail.com"
    email_to=email
    ses.send_email(
        Source = email_from,
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

```

