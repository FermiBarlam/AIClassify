__author__ = 'derog'

import arff

weather_data = arff.load(open("weather_nominal.arff"))


def OneRule(arff_data, if_attribute):
    index = 0
    if_index = 0

    for attribute in arff_data["attributes"]:
        if attribute[0] == if_attribute:
            if_index = index
        index += 1

    index = 0

    rules = {}
    for attribute in arff_data["attributes"]:
        if attribute[0] != if_attribute:
            # Generates the table with all the possible values
            count = {}
            for value in attribute[1]:
                for if_value in arff_data["attributes"][if_index][1]:
                    count[(value, if_value)] = 0

            for row in arff_data["data"]:
                count[(row[index], row[if_index])] += 1

            rules[attribute[0]] = []

            for value in attribute[1]:
                most_frequent = (None, 0)
                for if_value in arff_data["attributes"][if_index][1]:
                    if most_frequent[1] < count[(value, if_value)]:
                        most_frequent = (if_value, count[(value, if_value)])
                rules[attribute[0]].append((value, most_frequent[0], most_frequent[1]))
        index += 1

    best_rule = (None, None, 0)

    for rule in rules:
        win_sum = 0
        for condition in rules[rule]:
            win_sum += condition[2]

        if win_sum > best_rule[2]:
            best_rule = (rule, rules[rule], win_sum)

    def rule_apply(row):

        index = 0
        if_index = 0

        for attribute in arff_data["attributes"]:
            if attribute[0] == best_rule[0]:
                if_index = index
            index += 1

        value = row[if_index]

        for possible in best_rule[1]:
            if possible[0] == value:
                return possible[1]

    return rule_apply


Rule = OneRule(weather_data, 'play')
print(Rule(['rainy', 'mild', 'high', 'TRUE', 'no']))