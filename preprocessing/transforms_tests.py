from hackathon_test_transformer import test_transform_
from hackathon_transformers import transform_remove_cid, transform_remove_mail_header

cases = [
    ('[cid:image002.png@01d5f239.20062930]', ''),
    ('0 1 2 3 bern', '0 1 2 3 bern')
]
test_transform_(cases, transform_remove_cid)

only_header = 'from: six https://www.admin.ch>  sent: wednesday, november 28, 2018 2:17 pm to: _eda-dr https://www.admin.ch> cc: https://www.admin.ch> subject:'

cases = [
    (only_header, ''),
]
test_transform_(cases,transform_remove_mail_header)