 # coding=utf-8

label_set = [
  'Smart_Device___MDM', 'Benutzeranleitungen_Telefonie', 'Notebook', 'Outlook', 'Drucker___Multifunktionsgeräte_',
  'Standalone', 'Word', 'SAP_Online_Support', 'Interactive_Voice_Response__IVR_', 'Bestellungen', 'Intranet', 'eSysp___Biometrie',
  'Software_Center', 'Keepass', 'CH@WORLD', 'Informatiksicherheit', 'PDF_', 'Mitnahme_der_persönlichen_Daten_bei_Versetzung', 'Word|PDF_',
  'Datenaustausch', 'Smart_Device___MDM|Notebook', 'Standalone|Bestellungen', 'Satellitentelefonen', 'Elektronisch_Signieren', 'Smart_Device___MDM|Benutzeranleitungen_Telefonie',
  'Snipping_Tool', 'Office_Manager', 'Standalone|Benutzeranleitungen_Telefonie', 'Notebook|Bestellungen', 'Smart_Device___MDM|Informatiksicherheit',
  'Interactive_Voice_Response__IVR_|Benutzeranleitungen_Telefonie', 'Standalone|Notebook', 'Powerpoint', 'Software_Center|Informatiksicherheit',
  'Smart_Device___MDM|Outlook', 'Word|Software_Center', 'Smart_Device___MDM|Mitnahme_der_persönlichen_Daten_bei_Versetzung', 'Smart_Device___MDM|Bestellungen',
  'eSysp___Biometrie|Drucker___Multifunktionsgeräte_', 'Office_Manager|Word', 'Smart_Device___MDM|EDAInventory', 'Smart_Device___MDM|Mitnahme_der_persönlichen_Daten_bei_Versetzung|Intranet',
  'Smart_Device___MDM|eSysp___Biometrie|Notebook', 'EDAssist+|Informatiksicherheit', 'eSysp___Biometrie|EDAssist+', 'Word|eVera', 'Powerpoint|Outlook',
  'Standalone|Outlook', 'eVera', 'Mitnahme_der_persönlichen_Daten_bei_Versetzung|Informatiksicherheit', 'Mitnahme_der_persönlichen_Daten_bei_Versetzung|Outlook',
  'Informatiksicherheit|Outlook', 'Standalone|Keepass', 'Benutzeranleitungen_Telefonie|Informatiksicherheit', 'Notebook|Drucker___Multifunktionsgeräte_',
  'Word|Notebook', 'Notebook|EDAssist+', 'EDAssist+', 'Smart_Device___MDM|Word', 'Word|Outlook', 'CH@WORLD|Informatiksicherheit', 'Mitnahme_der_persönlichen_Daten_bei_Versetzung|OneNote|Intranet',
  'Office_Manager|Drucker___Multifunktionsgeräte_', 'Word|PDF_|Software_Center', 'Office_Manager|eSysp___Biometrie|eVera|Notebook|Drucker___Multifunktionsgeräte_|Software_Center|Benutzeranleitungen_Telefonie',
  'eSysp___Biometrie|Informatiksicherheit', 'Benutzeranleitungen_Telefonie|Outlook|Intranet', 'OneNote', 'CodX_PostOffice|Drucker___Multifunktionsgeräte_'
]

idx_to_labels_list = label_set  # list to look up the label indices
id2label = {k: v for k, v in enumerate(idx_to_labels_list)}
label2id = {v: k for k, v in enumerate(idx_to_labels_list)}

def get_id(label):
  return label2id[label]
print(get_id('Standalone|Outlook'))
