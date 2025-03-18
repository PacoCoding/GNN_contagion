#%%
import pandas as pd
import eikon as ek
from dotenv import load_dotenv
import os
load_dotenv()

eikon_key = os.getenv("EIKON_API_KEY")
ek.set_app_key(eikon_key)
#%%
#----------------------------------------
# Here We will get the rics using the SCREEN of Refinitiv, We get all the company that has as TRBC (The Refinitiv Business Classification): "55101010"
# that represents all the banks that are under the following subcategories: 
# [Banks (NEC), Corporate Banks, Retail & Mortgage Banks, Money Center Banks, Private Banks, Islamic banks]
#
# We also used the CURN setting in order to get all the values based on USD dollar in order to facilitate the following computations 
#
# Then we simply retrieve all the data using the builtin function get_data of Eikon that returns us a dataframe
#------------------------------------------
screener_exp = 'SCREEN(U(IN(Equity(active,public,primary))),IN(TR.TRBCIndustryCode,"55101010"),CURN=USD)'

ric_data, err = ek.get_data(screener_exp, ['TR.RIC'])
#%%
if err is None:
    bank_ric_list = ric_data['RIC'].dropna().tolist()  # Remove NaNs if any
    print(f"Retrieved {len(bank_ric_list)} bank RICs worldwide.")

    # Step 2: Retrieve Net Loans & Total Deposits for these banks
    if len(bank_ric_list) > 0:
        financial_data, err2 = ek.get_data(bank_ric_list, 
                                           fields=["TR.CompanyName",
                                                   "TR.F.LoansNetTot(Period=FY0)", 
                                                   "TR.F.DeposTot(Period=FY0)",
                                                   "TR.F.TotAssets(Period=FY0)",
                                                   "TR.F.TotLiab(Period=FY0)",
                                                   "TR.F.TotLiabEq(Period=FY0)",
                                                   "TR.F.InvstAssetsTot(Period=FY0)",
                                                   "TR.F.CashSTInvstTot(Period=FY0)",         
                                                   "TR.Tier1CapitalRatioMean(Period=FY1)",
                                                   "TR.RiskWeightedAssetsActual(Period=FY0)",
                                                   "TR.InvtrReturnOnEquity",
                                                   "TR.F.IntrIncLoansDepos(Period=FY0)",
                                                   "TR.F.CashSTDeposDueBanksTot(Period=FY0)",
                                                   "TR.F.LendLTDeposDueBanks(Period=FY0)",
                                                   "TR.F.DeposBankFinInstit(Period=FY0)"
                                                  ])
else:
    print("Error retrieving bank RICs:", err)

#financial_data['Equity'] = financial_data['Total Assets'] - financial_data['Total Liabilities']
#financial_data['Equity']
#%%

#%%
display(financial_data.dropna())