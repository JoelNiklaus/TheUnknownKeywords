import re

def transform_remove_cid(x):
    x = re.sub('\[cid\:.+\]', '', x)
    return x
def transform_remove_mail_header(x):
    x = re.sub('from\:.+subject\:', '', x, flags=re.IGNORECASE)
    return x