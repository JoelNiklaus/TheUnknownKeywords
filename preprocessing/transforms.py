import re

def transform_remove_cid(x):
    x = re.sub(r'\[cid\:.+\]', '', x)
    return x
def transform_remove_mail_header(x):
    x = re.sub(r'from\:.+subject\:', '', x, flags=re.IGNORECASE)
    return x
def transform_remove_inline_js(x):
    x = re.sub(r'\w\\\:\*.+\;\}', '', x)
    return x.strip()
def transform_remove_md5_hash(x):
    x = re.sub(r'MD5\:.{32}','',x)
    return x.strip()

def get_all_body_transforms():
    transforms = [
        transform_remove_cid,
        transform_remove_mail_header,
        transform_remove_inline_js
    ]
    return transforms