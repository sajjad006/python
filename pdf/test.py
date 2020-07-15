import tabula

url = "https://www.wbhealth.gov.in/uploaded_files/corona/Vacant_bed_status_as_on_02.07_.2020_.pdf"
# df = tabula.read_pdf(url, encoding='utf-8', stream=True, pages='all')
tabula.convert_into(url, "output.csv", output_format="csv", pages='all')
# print(df)