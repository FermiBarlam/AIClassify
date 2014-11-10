__author__ = 'derog'


def Discretitzaction(real_list, discrete_list, breaks=-1, cluster_size=3):
    nelements = len(real_list)
    if nelements != len(discrete_list):
        raise IndexError("The amount of elements in the differtent lists is not the same")

    # If we want to expand the code this categories will help us do so
    categories = 0

    if breaks == -1:
        keys = {}
        for element in discrete_list:
            keys[element] = 0
        categories = len(keys)
    else:
        categories = breaks

    # We should order both lists first thing, in order to do so
    # we will use a modified quicksort with a passing function for elements
    def quick_sort(a_list, ev_func):
        def partition(p, r):
            x = ev_func(a_list[r])
            i = p - 1
            for j in range(p, r):
                if ev_func(a_list[j]) <= x:
                    i += 1
                    a_list[i], a_list[j] = a_list[j], a_list[i]
            a_list[i + 1], a_list[r] = a_list[r], a_list[i + 1]
            return i + 1

        def _quick_sort(p, r):
            if p < r:
                q = partition(p, r)
                _quick_sort(p, q - 1)
                _quick_sort(q + 1, r)

        _quick_sort(0, len(a_list) - 1)

    fus_list = []
    for index in range(nelements):
        fus_list.append((real_list[index], discrete_list[index]))

    quick_sort(fus_list, lambda x: x[0])

    #Now that we are set, lets build the breaks we will set different rules and estimate the number of breaks from that
    breaks = []
    keys = {}
    for element in discrete_list:
        keys[element] = 0

    for index in range(nelements):
        keys[fus_list[index][1]] += 1
        if keys[fus_list[index][1]] == cluster_size:
            for element in discrete_list:
                keys[element] = 0
            breaks.append(index)

    def main_value(fus_list, init, end):
        max = (None, 0)
        keysp = {}
        for element in discrete_list:
            keysp[element] = 0
        for element in fus_list[init:end]:
            keysp[element[1]] += 1
            if keysp[element[1]] > max[1]:
                max = (element, keysp[element[1]])

        return element[1]

    #We build the intervals
    init_interval = 0
    intervals = []
    for index in range(len(breaks)):
        value = main_value(fus_list, init_interval, breaks[index])
        intervals.append((init_interval, breaks[index], value))
        init_interval = breaks[index] + 1

    #Now we stick up the intervals
    tinvervals = []
    tinvervals.append(intervals[0])
    for sub_interval in intervals:
        tinvervals.append((fus_list[sub_interval[0]][0], fus_list[sub_interval[1]][0], sub_interval[2]))
        if sub_interval[2] == tinvervals[-2][2]:
            tinvervals[-2] = (tinvervals[-2][0], fus_list[sub_interval[1]][0], tinvervals[-2][2])
            tinvervals.pop(len(tinvervals) - 1)

    #Now lets build the function rules from the tintervals
    class Apply_Rules():
        def __init__(self, sub_intervals):
            self.sub_intervals = sub_intervals

        def __str__(self):
            if len(self.sub_intervals) == 1:
                text = 'For any value we will return ' + sub_interval[0][2]
            elif len(self.sub_intervals) == 2:
                text = '(-inf,' + str((self.sub_intervals[0][1] + self.sub_intervals[1][0]) / 2) + ']->' + str(
                    self.sub_intervals[0][2]) + '\n'
                text += '(' + str((self.sub_intervals[0][1] + self.sub_intervals[1][0]) / 2) + ',+inf)->' + str(
                    self.sub_intervals[1][2])
            else:
                text = '(-inf,' + str((self.sub_intervals[0][1] + self.sub_intervals[1][0]) / 2) + ']->' + str(
                    self.sub_intervals[0][2]) + '\n'
                for index in range(len(sub_interval[1:-1])):  #Notice all index must be added 1
                    text += '(' + str((self.sub_intervals[index][1] + self.sub_intervals[index + 1][0]) / 2) + \
                            ',' + str(
                        (self.sub_intervals[index + 1][1] + self.sub_intervals[index + 2][0]) / 2) + ']->' + str(
                        self.sub_intervals[index + 1][2]) + '\n'
                text += '(' + str((self.sub_intervals[-2][1] + self.sub_intervals[-1][0]) / 2) + ',+inf)->' + str(
                    self.sub_intervals[-1][2])

            return text

        def __call__(self, *args, **kwargs):

            if len(self.sub_intervals) == 1:
                return sub_interval[0][2]
            elif len(self.sub_intervals) == 2:
                if args[0] <= (self.sub_intervals[0][1] + self.sub_intervals[1][0]) / 2:
                    return self.sub_intervals[0][2]
                else:
                    return self.sub_intervals[1][2]
            else:
                if args[0] <= (self.sub_intervals[0][1] + self.sub_intervals[1][0]) / 2:
                    return self.sub_intervals[0][2]
                for index in range(len(sub_interval[1:-1])):  #Notice all index must be added 1
                    if args[0] > (self.sub_intervals[index][1] + self.sub_intervals[index + 1][0]) / 2 and args[0] <= (
                        self.sub_intervals[index + 1][1] + self.sub_intervals[index + 2][0]) / 2:
                        return self.sub_intervals[index + 1][2]
                if args[0] > (self.sub_intervals[-2][1] + self.sub_intervals[-1][0]) / 2:
                    return self.sub_intervals[-1][2]

    Function = Apply_Rules(tinvervals)

    return Function


def NaiveBayes(arff_data, if_attribute):
    # First we determine the desired indexs we want to work with
    index = 0
    if_index = 0

    import math

    for attribute in arff_data["attributes"]:
        if attribute[0] == if_attribute:
            if_index = index
        index += 1

    # Build an statistic table with the relevant information
    index = 0
    rules = {}
    for attribute in arff_data["attributes"]:
        # Generates the table with all the possible values
        count = {}
        if isinstance(attribute[1], list):
            # Scenario for discrete data
            for value in attribute[1]:
                for if_value in arff_data["attributes"][if_index][1]:
                    count[(value, if_value)] = 0
            # Count the data
            for row in arff_data["data"]:
                count[(row[index], row[if_index])] += 1
        else:
            # Scenario for continuous data
            for if_value in arff_data["attributes"][if_index][1]:
                count[('mean', if_value)] = 0
                count[('n', if_value)] = 0
                count[('stddev', if_value)] = 0
            # Count the data
            for row in arff_data["data"]:
                count[('mean', row[if_index])] += row[index]
                count[('n', row[if_index])] += 1
            #Mean evaluation
            for outcome in arff_data["attributes"][if_index][1]:
                count[('mean', outcome)] /= count[('n', outcome)]
            for row in arff_data["data"]:
                count[('stddev', row[if_index])] += math.pow((row[index] - count[('mean', row[if_index])]), 2)
            #Standard Desviation
            for outcome in arff_data["attributes"][if_index][1]:
                count[('stddev', outcome)] /= (count[('n', outcome)] - 1)
                count[('stddev', outcome)] = math.sqrt(count[('stddev', outcome)])

        rules[attribute[0]] = count
        index += 1

    # Normalize discrete value, this code can be embbed up
    # TODO <Emb this code in the upper bucles>
    for attribute in arff_data["attributes"]:
        if isinstance(attribute[1], list):
            if_denom = 0
            for outcome in arff_data["attributes"][if_index][1]:
                if attribute[0] == if_attribute:
                    if_denom += rules[attribute[0]][outcome, outcome]
                else:
                    denom = 0
                    for value in attribute[1]:
                        denom += rules[attribute[0]][value, outcome]
                    for value in attribute[1]:
                        rules[attribute[0]][value, outcome] /= denom
            if attribute[0] == if_attribute:
                for outcome in arff_data["attributes"][if_index][1]:
                    rules[attribute[0]][outcome, outcome] /= if_denom

    #Chance of a continuous value given the mean and stddesv
    def normal_distribution_chance(mean, stddesv, value):
        return math.exp((-1) * ((math.pow((value - mean), 2)) / (2 * math.pow(stddesv, 2)))) / (
        math.sqrt(2 * math.pi) * stddesv)

    #Since in this scenario we have more info than an outcome we will make a class to return
    #TODO <Add inner properties so it remains clear what values are being used
    class Rule_Apply():
        def __call__(self, *args, **kwargs):

            row = args[0]
            chances = {}
            for outcome in arff_data["attributes"][if_index][1]:
                chances[outcome] = 1
                index = 0
                for attribute in arff_data["attributes"]:
                    if isinstance(attribute[1], list):
                        if index != if_index:
                            chances[outcome] *= rules[attribute[0]][row[index], outcome]
                        else:
                            chances[outcome] *= rules[attribute[0]][outcome, outcome]
                    else:
                        if index != if_index:
                            mean = rules[attribute[0]]['mean', outcome]
                            stddesv = rules[attribute[0]]['stddev', outcome]
                            chances[outcome] *= normal_distribution_chance(mean, stddesv, row[index])
                        else:
                            raise ValueError("Naive Bayes is not prepared for not discrete output")
                    index += 1

            if args[1] == True:

                sump = 0
                for outcome in arff_data["attributes"][if_index][1]:
                    sump += chances[outcome]
                for outcome in arff_data["attributes"][if_index][1]:
                    chances[outcome] /= sump

                return str(chances)

            else:
                biggest = (None, 0)
                for outcome in arff_data["attributes"][if_index][1]:
                    if chances[outcome] > biggest[1]:
                        biggest = (outcome, chances[outcome])
                return biggest[0]

    rule_class = Rule_Apply()
    return rule_class


def OneRule(arff_data, if_attribute):
    # Find what index we need to work with
    index = 0
    if_index = 0

    for attribute in arff_data["attributes"]:
        if attribute[0] == if_attribute:
            if_index = index
        index += 1

    # Build an statistic table with the relevant information
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

            # We only need to take the most common value
            rules[attribute[0]] = []
            for value in attribute[1]:
                most_frequent = (None, 0)
                for if_value in arff_data["attributes"][if_index][1]:
                    if most_frequent[1] < count[(value, if_value)]:
                        most_frequent = (if_value, count[(value, if_value)])
                rules[attribute[0]].append((value, most_frequent[0], most_frequent[1]))
        index += 1

    # Now we build up the rule that the function will follow
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


def main():
    # The data we will work with
    import arff

    weather_data = arff.load(open("weather_nominal.arff"))
    weather_data_real = arff.load(open("weather.arff"))

    # This returns a simple discrete prediction
    print("1R DECISION(Discrete):")
    Rule = OneRule(weather_data, 'play')
    print(Rule(['rainy', 'mild', 'high', 'TRUE', 'no']))

    # This returns some probabilities aside from a prediction
    print("NAIVEBAYES DECISION(Discrete):")
    SeconRule = NaiveBayes(weather_data, 'play')
    print(SeconRule(['sunny', 'cool', 'high', 'TRUE', 'no'], True))
    print(SeconRule(['sunny', 'cool', 'high', 'TRUE', 'no'], False))

    print("Discretitzation tools:")
    # To test the discrete algorithm we will provide different pairs of list
    #Two lists for temperature, discrete and continuous
    disc_temperature = []
    cont_temperature = []
    #Also for the humidity we work with two lists
    disc_humidity = []
    cont_humidity = []
    for element in weather_data['data']:
        disc_temperature.append(element[1])
        disc_humidity.append(element[2])

    for element in weather_data_real['data']:
        cont_temperature.append(element[1])
        cont_humidity.append(element[2])

    #Example for discretitzation of values following break rules
    print(Discretitzaction(cont_temperature, disc_temperature))
    print(Discretitzaction(cont_humidity, disc_humidity))

    #This returns some probabilities aside from a prediction
    print("NAIVEBAYES DECISION(Real Data):")
    SeconRule = NaiveBayes(weather_data_real, 'play')
    print(SeconRule(['sunny', 66, 90, 'TRUE', 'no'], True))
    print(SeconRule(['sunny', 66, 90, 'TRUE', 'no'], False))

if __name__ == '__main__':
    main()

