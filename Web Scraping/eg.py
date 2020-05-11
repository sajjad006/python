country_dict = {"USA": "United States of America", "UK": "United Kingdom", "UAE": "United Arab Emirates", 
                "S. Korea": "Korea South", "Czechia": "Czech Republic", "North Macedonia": "Macedonia", 
                "Ivory Coast": "Cote d'Ivoire", "DRC": "Democratic Republic of the Congo", "Taiwan": "Republic of China",
                "Réunion": "France", "Palestine": "Palestinian territories", "Congo": "Republic of the Congo",
                "Guinea-Bissau": "Guinea Bissau", "Faeroe Islands": "Faroe Islands", "Cabo Verde": "Cape Verde",
                "Eswatini": "Swaziland", "CAR": "Central African Republic", "Timor-Leste": "East Timor", "Curaçao": "Curacao",
                "St. Vincent Grenadines": "Saint Vincent and the Grenadines", "Turks and Caicos": "Turks and Caicos Islands",
                "British Virgin Islands": "Virgin Islands", "St. Barth": "Saint Barthelemy", "Caribbean Netherlands": "	Netherlands",
                "Saint Pierre Miquelon": "Saint Pierre and Miquelon"}

name = 'India'

if name in country_dict:
	print(country_dict[name])
else:
	print(False)