from __future__ import print_function
import argparse
import pprint
import requests
import sys
import urllib
import json
import datetime
import time
import os
import dateutil.parser
import logging
import math
import boto3

# This client code can run on Python 2.x or 3.x.  Your imports can be
# simpler if you only need one of those.
try:
    # For Python 3.0 and later
    from urllib.error import HTTPError
    from urllib.parse import quote
    from urllib.parse import urlencode
except ImportError:
    # Fall back to Python 2's urllib2 and urllib
    from urllib2 import HTTPError
    from urllib import quote
    from urllib import urlencode

import json
import datetime
import time
import os
import dateutil.parser
import logging
import math

# Yelp Fusion no longer uses OAuth as of December 7, 2017.
# You no longer need to provide Client ID to fetch Data
# It now uses private keys to authenticate requests (API Key)
# You can find it on
# https://www.yelp.com/developers/v3/manage_app
API_KEY = 'eJn78U82sS7oi8pzV3CVsn0tcw_hFu7sSjx22NuQebgm3L7dvEuwutZBU1L2YA4HRjzrdvUwuvNzMFt_RLVlyWE9qnTw4DGFJsYoMiqqBL3gHyqC1cZWxtIvFSS5XHYx'

# API constants, you shouldn't have to change these.
API_HOST = 'https://api.yelp.com'
SEARCH_PATH = '/v3/businesses/search'
BUSINESS_PATH = '/v3/businesses/'  # Business ID will come after slash.

# Defaults for our simple example.
DEFAULT_TERM = 'retaurants'
DEFAULT_LOCATION = 'New York, NY'
SEARCH_LIMIT = 3

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


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


def search(api_key, term, location, categories):
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
        'categories': categories.replace(' ', '+')
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


def query_api(term, location, categories):
    """Queries the API by the input values from the user.

    Args:
        term (str): The search term to query.
        location (str): The location of the business to query.
    """
    response = search(API_KEY, term, location, categories)
    businesses = response.get('businesses')

    if not businesses:
        print(u'No businesses for {0} in {1} found.'.format(term, location, categories))
        return

    res, name, addr, i = '', '', '', 1

    for b in businesses:
        name = b['name']
        addr = "".join(b['location']['address1'])
        res = res + str(i) + '. ' + name + ', located at ' + addr + '\n'
        i += 1
    return res

    # business_id = businesses[0]['id']

    # print(u'{0} businesses found, querying business info ' \
    #     'for the top result "{1}" ...'.format(
    #         len(businesses), business_id))
    # response = get_business(API_KEY, business_id)
    # pprint.pprint(response, indent=2)

    # print(u'Result for business "{0}" found:'.format(business_id))
    # # pprint.pprint(response, indent=2)
    # print(len(response))
    # print(response['name'])
    # print("".join(response['location']['display_address']))


def find(term, location, categories):
    res = ''
    try:
        res = query_api(term, location, categories)
    except HTTPError as error:
        sys.exit(
            'Encountered HTTP error {0} on {1}:\n {2}\nAbort program.'.format(
                error.code,
                error.url,
                error.read(),
            )
        )
    return res


"""
Lex Code Hook

"""


# --- Helpers that build all of the responses ---


def elicit_slot(session_attributes, intent_name, slots, slot_to_elicit, message):
    return {
        'sessionAttributes': session_attributes,
        'dialogAction': {
            'type': 'ElicitSlot',
            'intentName': intent_name,
            'slots': slots,
            'slotToElicit': slot_to_elicit,
            'message': message
        }
    }


def confirm_intent(session_attributes, intent_name, slots, message):
    return {
        'sessionAttributes': session_attributes,
        'dialogAction': {
            'type': 'ConfirmIntent',
            'intentName': intent_name,
            'slots': slots,
            'message': message
        }
    }


def close(session_attributes, fulfillment_state, message):
    response = {
        'sessionAttributes': session_attributes,
        'dialogAction': {
            'type': 'Close',
            'fulfillmentState': fulfillment_state,
            'message': message
        }
    }

    return response


def delegate(session_attributes, slots):
    return {
        'sessionAttributes': session_attributes,
        'dialogAction': {
            'type': 'Delegate',
            'slots': slots
        }
    }


# --- Helper Functions ---


def safe_int(n):
    """
    Safely convert n value to int.
    """
    if n is not None:
        return int(n)
    return n


def try_ex(func):
    """
    Call passed in function in try block. If KeyError is encountered return None.
    This function is intended to be used to safely access dictionary.

    Note that this function would have negative impact on performance.
    """

    try:
        return func()
    except KeyError:
        return None


def parse_int(n):
    try:
        return int(n)
    except ValueError:
        return float('nan')


def isvalid_cuisine(cuisine):
    cuisines = ['japanese', 'indian', 'american', 'chinese', 'france', 'italian', 'korean']
    return cuisine.lower() in cuisines


def isvalid_city(city):
    valid_cities = ['mahattan', 'new york', 'los angeles', 'chicago', 'houston', 'philadelphia', 'phoenix',
                    'san antonio','san diego', 'dallas', 'san jose', 'austin', 'jacksonville', 'san francisco', 'indianapolis',
                    'columbus', 'fort worth', 'charlotte', 'detroit', 'el paso', 'seattle', 'denver', 'washington dc',
                    'memphis', 'boston', 'nashville', 'baltimore', 'portland']
    return city.lower() in valid_cities


def isvalid_room_type(room_type):
    room_types = ['queen', 'king', 'deluxe']
    return room_type.lower() in room_types


def isvalid_date(date):
    try:
        dateutil.parser.parse(date)
        return True
    except ValueError:
        return False


def build_validation_result(isvalid, violated_slot, message_content):
    return {
        'isValid': isvalid,
        'violatedSlot': violated_slot,
        'message': {'contentType': 'PlainText', 'content': message_content}
    }


def validate_dining(slots):
    location = try_ex(lambda: slots['Location'])
    dining_date = try_ex(lambda: slots['DiningDate'])
    dining_time = try_ex(lambda: slots['DiningTime'])
    # nights = safe_int(try_ex(lambda: slots['Nights']))
    cuisine = try_ex(lambda: slots['Cuisine'])
    number = safe_int(try_ex(lambda: slots['Number']))

    if location and not isvalid_city(location):
        return build_validation_result(
            False,
            'Location',
            'We currently do not support {} as a valid destination.  Can you try a different city?'.format(location)
        )

    if dining_date:
        if not isvalid_date(dining_date):
            return build_validation_result(False, 'DiningDate',
                                           'I did not understand your dining date.  What date do you like to eat?')
        if datetime.datetime.strptime(dining_date, '%Y-%m-%d').date() < datetime.date.today():
            return build_validation_result(False, 'DiningDate',
                                           'Dining time must be scheduled from today onwards. Can you try a different date?')

    if dining_time is not None:
        if len(dining_time) != 5:
            # Not a valid time; use a prompt defined on the build-time model.
            return build_validation_result(False, 'DiningTime',
                                           'I did not recognize that, what time would you like to eat?')

        hour, minute = dining_time.split(':')
        hour = parse_int(hour)
        minute = parse_int(minute)
        if math.isnan(hour) or math.isnan(minute):
            # Not a valid time; use a prompt defined on the build-time model.
            return build_validation_result(False, 'DiningTime', None)

    if number is not None and (number < 1 or number > 30):
        return build_validation_result(
            False,
            'Number',
            'Restaurants only for 1 to 100 people.  How many people do you have?'
        )

    if cuisine and not isvalid_cuisine(cuisine):
        return build_validation_result(False, 'Cuisine',
                                       'I did not recognize that cuisine type.  What cuisine would you like to try?')

    return {'isValid': True}


def generate_restaurant(location, cuisine, number, dining_date, dining_time):
    return find('retaurants', location, cuisine)


""" --- Functions that control the bot's behavior --- """


def find_dining_place(intent_request):
    """
    Performs dialog management and fulfillment for finding a dining place.

    Beyond fulfillment, the implementation for this intent demonstrates the following:
    1) Use of elicitSlot in slot validation and re-prompting
    2) Use of sessionAttributes to pass information that can be used to guide conversation
    """

    location = try_ex(lambda: intent_request['currentIntent']['slots']['Location'])
    dining_date = try_ex(lambda: intent_request['currentIntent']['slots']['DiningDate'])
    dining_time = try_ex(lambda: intent_request['currentIntent']['slots']['DiningTime'])
    number = safe_int(try_ex(lambda: intent_request['currentIntent']['slots']['Number']))
    cuisine = try_ex(lambda: intent_request['currentIntent']['slots']['Cuisine'])
    email = try_ex(lambda: intent_request['currentIntent']['slots']['Email'])
    session_attributes = intent_request['sessionAttributes'] if intent_request['sessionAttributes'] is not None else {}

    # Load confirmation history and track the current reservation.
    suggestion = json.dumps({
        'ReservationType': 'Dining',
        'Location': location,
        'Cuisine': cuisine,
        'DiningDate': dining_date,
        'DiningTime': dining_time,
        'Number': number,
        'Email': email
    })

    session_attributes['currentReservation'] = suggestion

    if intent_request['invocationSource'] == 'DialogCodeHook':
        # Validate any slots which have been specified.  If any are invalid, re-elicit for their value
        validation_result = validate_dining(intent_request['currentIntent']['slots'])
        if not validation_result['isValid']:
            slots = intent_request['currentIntent']['slots']
            slots[validation_result['violatedSlot']] = None

            return elicit_slot(
                session_attributes,
                intent_request['currentIntent']['name'],
                slots,
                validation_result['violatedSlot'],
                validation_result['message']
            )

        # Otherwise, let native DM rules determine how to elicit for slots and prompt for confirmation.  Pass price
        # back in sessionAttributes once it can be calculated; otherwise clear any setting from sessionAttributes.
        if location and dining_date and dining_time and number and cuisine and email:
            # The price of the hotel has yet to be confirmed.
            restaurant = generate_restaurant(location, cuisine, number, dining_date, dining_time)
            sendSQSmessage(location, cuisine, number, dining_date, dining_time, email)
            session_attributes['currentSuggestion'] = restaurant
        else:
            try_ex(lambda: session_attributes.pop('currentSuggestion'))

        session_attributes['currentReservation'] = suggestion
        return delegate(session_attributes, intent_request['currentIntent']['slots'])

    # Booking the hotel.  In a real application, this would likely involve a call to a backend service.
    logger.debug('findRestaurant under={}'.format(suggestion))

    try_ex(lambda: session_attributes.pop('currentSuggestion'))
    try_ex(lambda: session_attributes.pop('currentReservation'))
    # session_attributes['lastConfirmedReservation'] = suggestion

    return close(
        session_attributes,
        'Fulfilled',
        {
            'contentType': 'PlainText',
            'content': 'Okay, Hope you will like it. Please let me know if you would like to find another place.'

        }

    )


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
    except IOError:
        print("Error!")
    else:
        print("Succeed!")

    print(response['MessageId'])


def greet_intent(intent_request):
    return close(
        None,
        'Fulfilled',
        {
            'contentType': 'PlainText',
            'content': 'Hi there, how can I help?'
        }
    )


def thank_intent(intent_request):
    return close(
        None,
        'Fulfilled',
        {
            'contentType': 'PlainText',
            'content': 'You are welcome!'
        }
    )


# --- Intents ---


def dispatch(intent_request):
    """
    Called when the user specifies an intent for this bot.
    """

    logger.debug(
        'dispatch userId={}, intentName={}'.format(intent_request['userId'], intent_request['currentIntent']['name']))

    intent_name = intent_request['currentIntent']['name']

    # Dispatch to your bot's intent handlers
    if intent_name == 'DiningSuggestionsIntent':
        return find_dining_place(intent_request)
    elif intent_name == 'GreetingIntent':
        return greet_intent(intent_request)
    elif intent_name == 'ThankYouIntent':
        return thank_intent(intent_request)

    raise Exception('Intent with name ' + intent_name + ' not supported')


# --- Main handler ---


def lambda_handler(event, context):
    """
    Route the incoming request based on intent.
    The JSON body of the request is provided in the event slot.
    """
    # By default, treat the user request as coming from the America/New_York time zone.
    os.environ['TZ'] = 'America/New_York'
    time.tzset()
    logger.debug('event.bot.name={}'.format(event['bot']['name']))

    return dispatch(event)
