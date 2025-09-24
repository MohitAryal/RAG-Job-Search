import re

def preprocess_job_location(location: str) -> dict:
    """
    Clean and normalize job location strings.
    Returns a structured dictionary.
    """

    # Return all empty if location is of float datatype (i.e. it's Null)
    if isinstance(location, float):
      return {"cities": [],
        "states": [],
        "countries": [],
        "is_remote": []
      }

    raw = location.strip()


    # Detect remote/flexible flag
    is_remote = any(keyword in raw.lower() for keyword in ["remote", "flexible"])

    # Remove "Flexible / Remote" for parsing cities
    cleaned = re.sub(r"(?i)flexible\s*/?\s*remote", "", raw).strip(", ")

    cities, states, countries = [], [], []

    # US format: City, ST
    if re.match(r"^[A-Za-z\s]+,\s*[A-Z]{2}$", cleaned):
        city, state = cleaned.split(",")
        cities.append(city.strip())
        states.append(state.strip())
        countries.append("USA")

    cleaned_split = [c.strip() for c in cleaned.split(',')]
    cleaned_len = len(cleaned_split)

    # Country only
    if cleaned_len == 1:
      countries.append(cleaned)

    # City, state, city, state format
    elif cleaned_len > 3:
      cities = cleaned_split[::2]
      states = cleaned_split[1::2]
      countries = 'USA'

    else:
      # City, Country format
      if cleaned_len == 2:
        city, country = cleaned_split

      # City, state, country format
      elif cleaned_len == 3:
        city, state, country = cleaned_split
        states.append(state)

      cities.append(city)
      countries.append(country)

    return {"cities": list(set(cities)),
        "states": list(set(states)),
        "countries": list(set(countries)),
        "is_remote": is_remote
        }