To find a good solution we tried to cluster the emails, to get good results we would need to properly clean the data, removing all names, places and other irrelevant data. Therefore our solution is in theoretical nature.

Our solution assigns labels based on the mentioned problems in the
email bodies. Some of the existing labels need to be splitted to fit in the new support labels.

Below is an example for new labels. We reduced the amount of labels from 55 to 21 Labels.

New Labels:

* Orders:
  * Hardware -> Orders for new hardware
    * EDA_S_Order Management
    * EDA_S_Mobile Kommunikation
    * EDA_S_APS_PC
    * EDA_S_APS_Monitor
  * Software -> Orders for new software
    * EDA_S_Order Management
    * EDA_S_Zusätzliche Software
    * EDA_S_APS_OS_BasisSW
* Issues:
  * Hardware -> Issues with current hardware
    * EDA_S_Peripheriegeräte
    * EDA_S_APS_PC
    * EDA_S_APS_Monitor
  * Software -> Issues with current software
    * EDA_S_Büroautomation
    * EDA_S_Benutzerunterstützung
* Workplace:
  * UserManagement -> New Accounts and Issues with old accounts
    * EDA_S_Backup & Restore
    * EDA_S_Benutzerunterstützung
    * EDA_S_BA_2FA
    * EDA_S_BA_Account
    * EDA_S_BA_Datenablage
  * Mailing -> Issues with outlook
    * EDA_S_Benutzerunterstützung
    * EDA_S_BA_Mailbox
  * Remote_Access
    * EDA_S_BA_RemoteAccess
  * Printers
    * EDA_S_Arbeitsplatzdrucker
    * EDA_S_Netzdrucker
    * EDA_S_Peripheriegeräte
  * Telephone
    * EDA_S_BA_UCC_Benutzertelefonie
    * EDA_S_Peripheriegeräte
* Networking and Security:
  * Domestic -> Issues in Switzerland
    * EDA_S_Netzwerk Inland
  * Foreign -> Issues in foreign offices
    * EDA_S_BA_ServerAusland
    * EDA_S_Netzwerk Ausland
  * transmissions -> Communication between offices
    * EDA_S_IT Sicherheit
    * EDA_S_Betrieb Übermitttlungssysteme
* Direct Support:
  * Requests: -> Requests for direct support
    * EDA_S_Benutzerunterstützung
    * EDA_S_Raumbewirtschaftung
* General:
  * querstions -> Everything else with can't be assigned a label
    * EDA_ANW_Reisehinweise
  * issues -> Everything else with can't be assigned a label
    * _Pending





EDA_S_Order Management                  224 Orders
EDA_S_Mobile Kommunikation              201 
EDA_S_Netzdrucker                       114
EDA_S_BA_UCC_Benutzertelefonie          147
EDA_S_Zusätzliche Software               83 Orders
EDA_S_Netzwerk Ausland                   76
EDA_S_Betrieb Übermitttlungssysteme      75
EDA_S_Benutzerunterstützung              73
EDA_S_Peripheriegeräte                   64
EDA_S_IT Sicherheit                      54
EDA_S_Raumbewirtschaftung                44
EDA_S_Büroautomation                     12
EDA_S_Netzwerk Inland                     7
EDA_S_Backup & Restore                    3
EDA_S_Arbeitsplatzdrucker                 1

EDA_S_BA_Mailbox                        216
EDA_S_BA_2FA    EDA_S_BA_Account                        100
EDA_S_BA_Account                         80
EDA_S_BA_Datenablage                     34
EDA_S_BA_RemoteAccess                    29
EDA_S_BA_Internetzugriff                 22
EDA_S_BA_ServerAusland                   14


EDA_S_APS_OS_BasisSW                    208
EDA_S_APS_PC                            149
EDA_S_APS_Monitor                        99
EDA_S_APS_Peripherie                     76
EDA_S_BA_UCC_IVR                         69
EDA_S_APS                                15


EDA_ANW_SysP eDoc                       173
EDA_ANW_Intranet/Collaboration EDA       90
EDA_ANW_SAP Services                     75
EDA_ANW_CH@World (MOSS)                  57
EDA_ANW_ZACWEB                           43
EDA_ANW_DMS Fabasoft eGov Suite          15
EDA_ANW_ARS Remedy                       13
EDA_ANW_MOVE!                            11
EDA_ANW_Internet EDA                     10
EDA_ANW_at Honorarvertretung              9
EDA_ANW_Plato-HH                          7
EDA_ANW_Zentrale Raumreservation EDA      4
EDA_ANW_IAM Tool EDA                      3
EDA_ANW_FDFA Security App                 3
EDA_ANW_Office Manager                    3
EDA_ANW_ITDoc Sharepoint                  3
EDA_ANW_EDA PWC Tool                      2
EDA_ANW_EDAContacts                       2
EDA_ANW_EDAssist+                         2
EDA_ANW_NOS:4                             2
EDA_ANW_ORBIS                             2
EDA_ANW_ARIS (EDA Scout)                  1
EDA_ANW_CodX PostOffice                   1
EDA_ANW_Zeiterfassung SAP                 1
EDA_ANW_eVERA                             1
EDA_ANW_Reisehinweise                     1

_Pending                                 22
