import json
import finnhub
import pandas as pd
finnhub_client = finnhub.Client(api_key="ctblghpr01qvslqur8k0ctblghpr01qvslqur8kg")

json_object = finnhub_client.company_news('NDAQ', _from="2024-06-10", to="2024-12-10")
json_file = json.dumps(json_object, indent=4)

with open("news.json", "w") as file:
    file.write(json_file)