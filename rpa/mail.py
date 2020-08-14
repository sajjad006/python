from __future__ import print_function
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

from email.mime.text import MIMEText

import os
import re
import base64
import email
import fileinput
import subprocess
from num2words import num2words

def data_encoder(text):
    if len(text)>0:
        message = base64.urlsafe_b64decode(text)
        message = str(message, 'utf-8')
        message = email.message_from_string(message)
    return message


def readMessage(content)->str:
    message = None
    if "data" in content['payload']['body']:
        message = content['payload']['body']['data']
        message = data_encoder(message)
    elif "data" in content['payload']['parts'][0]['body']:
        message = content['payload']['parts'][0]['body']['data']
        message = data_encoder(message)
    else:
        # print("body has no data.")
        pass
    return message


def ListMessagesWithLabels(service, user_id, label_ids=[]):
  # List all Messages of the user's mailbox with label_ids applied.
  try:
    response = service.users().messages().list(userId=user_id,
                                               labelIds=label_ids).execute()
    messages = []
    if 'messages' in response:
      messages.extend(response['messages'])

    while 'nextPageToken' in response:
      page_token = response['nextPageToken']
      response = service.users().messages().list(userId=user_id,
                                                 labelIds=label_ids,
                                                 pageToken=page_token).execute()
      messages.extend(response['messages'])

    return messages
  except:
    print('An error occurred')


def GetMessage(service, user_id, msg_id):
  #Get a Message with given ID.
  try:
    message = service.users().messages().get(userId=user_id, id=msg_id).execute()

    # print(f'Message snippet: {message['snippet']}')

    return message
  except:
    print('An error occurred')


def SendMessage(service, user_id, message):
    # Send an email message.
    try:
        message = (service.users().messages().send(userId=user_id, body=message).execute())
        # print 'Message Id: %s' % message['id']
        return message
    except Exception as e:
        print('An error occurred here', e)


def CreateMessage(sender, to, subject, message_text):
    # Create a message for an email.

    message = MIMEText(message_text)
    message['to'] = to
    message['from'] = sender
    message['subject'] = subject
    return {'raw': base64.urlsafe_b64encode(message.as_string().encode()).decode()}


def validatePrograms(count):
    results = {}
    for i in range(1, count+1):
        # Add String[] args to main  (if absent)
        with fileinput.FileInput(f'{num2words(i)}.txt', inplace=True) as file:
            for line in file:
                print(line.replace('main()', 'main(String[] args)'), end='')

        # get class name
        with open(f'{num2words(i)}.txt', 'r+') as file:
            first = file.readline()
            class_name = first.replace('{', '').split(' ')[-1].strip()
            
        os.rename(f'{num2words(i)}.txt', f'{class_name}.java')
        result_code = os.system(f'javac {class_name}.java')

        results[num2words(i)] = 0

        if result_code == 0:
            output = bytes.decode(subprocess.check_output(f'java {class_name}', shell=True))

            if '452.448' in output or '75.408'  in output:
                if '452.448' in output and '75.408'  in output:
                    results[num2words(i)]  = 5
                else:
                    results[num2words(i)]  = 2.5
            elif '42.5' in output:
                results[num2words(i)] = 5
            elif '156.25' in output or '50' in output:
                if '156.25' in output and '50' in output:
                    results[num2words(i)] = 5
                else:
                    results[num2words(i)] = 2.5
            elif open(f'{class_name}.java', 'r+').read().find('+') != -1:
                if open(f'{class_name}.java', 'r+').read().find('-') != -1:
                    nums = re.findall('= \d+;', open(f'{class_name}.java', 'r+').read())
                    if len(nums) < 3:
                        nums = re.findall('=\d+;', open(f'{class_name}.java', 'r+').read())
                    
                    nums = [int(re.findall('\d+', x)[0]) for x in nums]
                    print(nums)
                    try:
                        nums.remove(100)
                    except:
                        pass

                    if str(100 - sum(nums[:3])) in output:
                        results[num2words(i)] = 5

            os.remove(f'{class_name}.class')
            os.remove(f'{class_name}.java')
        else:
            os.remove(f'{class_name}.java')

    return results


# If modifying these scopes, delete the file token.pickle.
# SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
SCOPES = ['https://mail.google.com/']

def main():
    creds = None
    
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('gmail', 'v1', credentials=creds)

    # Call the Gmail API
    # Get Unread Mails...
    messages = ListMessagesWithLabels(service, "me", ['UNREAD'])

    # Iterate through the messages..
    for message in messages:
        m = GetMessage(service, "me", message['id'])
        headers = m['payload']['headers']
    
        for header in headers:
            if header['name'] == 'From':
                msg_from = header['value']
            elif header['name'] == 'Subject':
                msg_subject = header['value']

        msg_body = str(readMessage(m)).strip()

        # Skip to next interation if msg is not from class 9 student
        # if not '9' in msg_subject:
        #     continue

        c, x = 1, 0

        for part in m['payload']['parts']:
          if part['filename']:

            file_name = part['filename']
            ext = file_name.split('.')[-1]

            if ext != 'txt':
                continue
            else:
                x+=1

            attachment = service.users().messages().attachments().get(userId='me', messageId=message['id'], id=part['body']['attachmentId']).execute()
            file_data = base64.urlsafe_b64decode(attachment['data'].encode('UTF-8'))
            
            name = num2words(c)+'.txt'
            path = ''.join(['', name])

            f = open(path, 'wb')
            f.write(file_data)
            f.close()

            c+=1

        if x <  4:
            response = f"Hi,\nYou have submitted only {x} out of 4 programs. Please resend all your programs in correct order again.\nN.B: All your programs must have a .txt extension"
        else:
            msg_from = msg_from.replace('<', '').replace('>', '').split(' ')[-1]

            result = validatePrograms(x)
            total = sum(result.values())
            response = f"Your total score = {total}/20\nProgram 1: {result['one']}\nProgram 2: {result['two']}\nProgram 3: {result['three']}\nProgram 4: {result['four']}"

        response += '\n\nThis is a computer generated email.'

        print(response)
        # message_reply = CreateMessage("me", msg_from, "Computer Assignment Result", response)
        # message_send = SendMessage(service, "me", message_reply)
        
        # print(message_send)
        
if __name__ == '__main__':
    main()