label_set = ['EDA_ANW_ARIS (EDA Scout)', 'EDA_ANW_ARS Remedy', 'EDA_ANW_CH@World (MOSS)', 'EDA_ANW_CodX PostOffice',
             'EDA_ANW_DMS Fabasoft eGov Suite', 'EDA_ANW_EDA PWC Tool', 'EDA_ANW_EDAContacts', 'EDA_ANW_EDAssist+',
             'EDA_ANW_FDFA Security App', 'EDA_ANW_IAM Tool EDA', 'EDA_ANW_ITDoc Sharepoint', 'EDA_ANW_Internet EDA',
             'EDA_ANW_Intranet/Collaboration EDA', 'EDA_ANW_MOVE!', 'EDA_ANW_NOS:4', 'EDA_ANW_ORBIS',
             'EDA_ANW_Office Manager', 'EDA_ANW_Plato-HH', 'EDA_ANW_Reisehinweise', 'EDA_ANW_SAP Services',
             'EDA_ANW_SysP eDoc', 'EDA_ANW_ZACWEB', 'EDA_ANW_Zeiterfassung SAP', 'EDA_ANW_Zentrale Raumreservation EDA',
             'EDA_ANW_at Honorarvertretung', 'EDA_ANW_eVERA', 'EDA_S_APS', 'EDA_S_APS_Monitor', 'EDA_S_APS_OS_BasisSW',
             'EDA_S_APS_PC', 'EDA_S_APS_Peripherie', 'EDA_S_Arbeitsplatzdrucker', 'EDA_S_BA_2FA', 'EDA_S_BA_Account',
             'EDA_S_BA_Datenablage', 'EDA_S_BA_Internetzugriff', 'EDA_S_BA_Mailbox', 'EDA_S_BA_RemoteAccess',
             'EDA_S_BA_ServerAusland', 'EDA_S_BA_UCC_Benutzertelefonie', 'EDA_S_BA_UCC_IVR', 'EDA_S_Backup & Restore',
             'EDA_S_Benutzerunterstützung', 'EDA_S_Betrieb Übermitttlungssysteme', 'EDA_S_Büroautomation',
             'EDA_S_IT Sicherheit', 'EDA_S_Mobile Kommunikation', 'EDA_S_Netzdrucker', 'EDA_S_Netzwerk Ausland',
             'EDA_S_Netzwerk Inland', 'EDA_S_Order Management', 'EDA_S_Peripheriegeräte', 'EDA_S_Raumbewirtschaftung',
             'EDA_S_Zusätzliche Software', '_Pending']
idx_to_labels_list = label_set  # list to look up the label indices
id2label = {k: v for k, v in enumerate(idx_to_labels_list)}
label2id = {v: k for k, v in enumerate(idx_to_labels_list)}

def get_id(label):
  return label2id[label]