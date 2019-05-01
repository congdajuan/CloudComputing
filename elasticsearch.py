# -*- coding: utf-8 -*-
from elasticsearch import Elasticsearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3
from elasticsearch import helpers, Elasticsearch
import csv

host = 'search-my-ml-restaurants-mnncskxhgpqmz6psyq7sevxauu.us-east-1.es.amazonaws.com' # For example, my-test-domain.us-east-1.es.amazonaws.com
region = 'us-east-1' # e.g. us-west-1

service = 'es'
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service)

es = Elasticsearch(
    hosts = [{'host': host, 'port': 443}],
    http_auth = awsauth,
    use_ssl = True,
    verify_certs = True,
    connection_class = RequestsHttpConnection
)


res=[]
with open('/Users/cong/Downloads/FILE_3.csv') as f:
    reader = csv.DictReader(f)
    for line in reader:
        # print(line)
        if line['Label'] == '1':
            res.append({ "Business_ID" : line["Business_ID"], "Cuisine_Type" : line["Cuisine_Type"], "Score" : line["Score"],"Label" : line["Label"]})
    # print(res)
    structured_json_body = ({
        "_op_type": "index",
        "_index":'prediction' ,  # index name Twitter
        "_type": 'prediction',  # type is tweet
        "_id": doc['Business_ID'],  # id of the tweet
        "_source": doc} for doc in res)
    helpers.bulk(es, structured_json_body, index='Prediction', doc_type='_doc')


#
# {
#     "Business_ID" :  "xxxx",
#     "Cuisine_Type" :   "Chinese",
#     "Score" :         0.05,
#     "Label" :       1
# }
