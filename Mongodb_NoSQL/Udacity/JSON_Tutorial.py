# Daniel Dixey
# 9/5/2015

# Each Question is defined as One Function

# To experiment with this code freely you will have to run this code locally.
# We have provided an example json output here for you to look at,
# but you will not be able to run any queries through our UI.

import json
import requests
import time

BASE_URL = "http://musicbrainz.org/ws/2/"
ARTIST_URL = BASE_URL + "artist/"

query_type = {"simple": {},
              "atr": {"inc": "aliases+tags+ratings"},
              "aliases": {"inc": "aliases"},
              "releases": {"inc": "releases"}}


def query_site(url, params, uid="", fmt="json"):
    params["fmt"] = fmt
    r = requests.get(url + uid, params=params)
    # print "requesting", r.url

    if r.status_code == requests.codes.ok:
        return r.json()
    else:
        r.raise_for_status()


def query_by_name(url, params, name):
    params["query"] = "artist:" + name
    return query_site(url, params)


def pretty_print(data, indent=4):
    if isinstance(data, dict):
        print json.dumps(data, indent=indent, sort_keys=True)
    else:
        print data


def main():
    results = query_by_name(ARTIST_URL, query_type["simple"], "Nirvana")
    pretty_print(results)

    artist_id = results["artists"][1]["id"]
    print "\nARTIST:"
    pretty_print(results["artists"][1])

    artist_data = query_site(ARTIST_URL, query_type["releases"], artist_id)
    releases = artist_data["releases"]
    print "\nONE RELEASE:"
    pretty_print(releases[0], indent=2)
    release_titles = [r["title"] for r in releases]

    print "\nALL TITLES:"
    for t in release_titles:
        print t

# Question 1: How many bands named "First Aid Kit"?


def Question1():
    results = query_by_name(ARTIST_URL, query_type["simple"], "FIRST AID KIT")
    # pretty_print(results)
    counter = 0
    for i in range(0, len(results["artists"])):
        if results["artists"][i]["name"] == "First Aid Kit":
            counter += 1
    print('Number of Artists called: First Aid Kit = %d') % (counter)

# Question 2: Begin_Area name for Queen?


def Question2():
    results = query_by_name(ARTIST_URL, query_type["simple"], "Queen")
    # pretty_print(results)
    print('Begin Area name for Queen: %s') % (
        results["artists"][0]["begin-area"]["name"])

# Question 3: Spanish Alias for Beatles?


def Question3():
    results = query_by_name(ARTIST_URL, query_type["simple"], "Beatles")
    # pretty_print(results)
    for table in results["artists"][0]['aliases']:
        if table['locale'] == 'es':
            print('Spanish Alias for Beatles? %s') % (table['name'])

# Question 4: Nirvana disambiguation?


def Question4():
    results = query_by_name(ARTIST_URL, query_type["simple"], "Nirvana")
    # pretty_print(results)
    print('Nirvana disambiguation? %s') % (
        results["artists"][0]["disambiguation"])

# Questions 5: When was One Direction formed?


def Question5():
    results = query_by_name(ARTIST_URL, query_type["simple"], "One Direction")
    # pretty_print(results)
    print('When was One Direction formed? %s') % (
        results["artists"][0]["life-span"]["begin"])


if __name__ == '__main__':
    # Example Script
    # main()
    # # Question 1: How many bands named "First Aid Kit"?
    Question1()
    time.sleep(5)
    # Question 2: Begin_Area name for Queen?
    Question2()
    time.sleep(5)
    # Question 3: Spanish Alias for Beatles?
    Question3()
    time.sleep(5)
    # Question 4: Nirvana disambiguation?
    Question4()
    time.sleep(5)
    # Questions 5: When was One Direction formed?
    Question5()
