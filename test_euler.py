import exposure_euler as ex_eu
from concurrent.futures import ProcessPoolExecutor

print('Recognized')

def process_country(cty):
    """Wrapper function to process a single country."""
    return ex_eu.init_TC_exp(country=cty)

if __name__ == "__main__":
    # Define parameters
    print('Started')
    countries = [242, 388, 192, 626]  # List of country codes

    # Use ProcessPoolExecutor to parallelize across countries
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_country, countries))

    print('Job done')