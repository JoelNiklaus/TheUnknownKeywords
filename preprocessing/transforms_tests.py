# coding=utf-8
from test_transforms import test_transform_
from transforms import transform_remove_cid, transform_remove_mail_header, transform_remove_inline_js

cases = [
    ('[cid:image002.png@01d5f239.20062930]', ''),
    ('0 1 2 3 bern', '0 1 2 3 bern')
]
test_transform_(cases, transform_remove_cid)

only_header = 'from: six https://www.admin.ch>  sent: wednesday, november 28, 2018 2:17 pm to: _eda-dr https://www.admin.ch> cc: https://www.admin.ch> subject:'
full_msg = 'zur bearbeitung  gruss  marco reuter lernender kaufmann 3. lehrjahr  eidgenössisches departement für auswärtige angelegenheiten eda direktion für ressourcen dr informatik eda / it-governance und projekte / finanzen, controlling und administration  freiburgstrasse 0 1 2 3 , a 0 1 2 3 , 0 1 2 3 bern, schweiz tel. +0 1 2 3 https://www.admin.ch/>    from: dütschler https://www.admin.ch> sent: tuesday, february https://www.admin.ch> subject: acrobat-lizenz?  guten tag,  mein acrobat pro zeigt die meldung im anhang - können ihr schauen, dass mir der nicht abgestellt wird? merci!  liebe grüsse'
trans_msg = 'zur bearbeitung  gruss  marco reuter lernender kaufmann 3. lehrjahr  eidgenössisches departement für auswärtige angelegenheiten eda direktion für ressourcen dr informatik eda / it-governance und projekte / finanzen, controlling und administration  freiburgstrasse 0 1 2 3 , a 0 1 2 3 , 0 1 2 3 bern, schweiz tel. +0 1 2 3 https://www.admin.ch/>     acrobat-lizenz?  guten tag,  mein acrobat pro zeigt die meldung im anhang - können ihr schauen, dass mir der nicht abgestellt wird? merci!  liebe grüsse'
only_header_2 = 'From: rausis https://www.admin.ch>> sent: wednesday, october 30, 2019 9:50 am to: ndiaye christian david https://www.admin.ch>> subject:'

cases = [
    (only_header, ''),
    (only_header_2, ''),
    (full_msg, trans_msg)
]
test_transform_(cases,transform_remove_mail_header)

s = 'v\:* {behavior:url(#default#vml);} o\:* {behavior:url(#default#vml);} w\:* {behavior:url(#default#vml);} .shape {behavior:url(#default#vml);} '

cases = [
    ('v\:* {behavior:url(#default#vml);}', ''),
    (s, '')
]

test_transform_(cases, transform_remove_inline_js)