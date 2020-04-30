# -*- coding: utf-8 -*-
import gensim
train_df = pd.read_csv('train.tsv', delimiter='\t')

train_df.head()

params = {'C': [1.0, 0.1], 'fit_intercept': [False], 'ngram_range': [(1, 3)], 'max_features': [10000, 5000],
          'max_iter': [1000, 500], 'use_tf_idf': [True, False], 'analyzer': ['word']}


class TrainAgent():
    def __init__(self, combination):
        self.combination = combination

    def get_performance(self, X_train, X_test, y_train, y_test):
        combination = self.combination

        C = combination['C']
        fit_intercept = combination['fit_intercept']
        max_features = combination['max_features']
        ngram_range = combination['ngram_range']
        max_iter = combination['max_iter']
        use_tf_idf = combination['use_tf_idf']
        analyzer = combination['analyzer']

        if use_tf_idf:
            vect = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, analyzer=analyzer,
                                   dtype=np.float32)
        else:
            vect = CountVectorizer(max_features=max_features, ngram_range=ngram_range, analyzer=analyzer,
                                   dtype=np.float32)

        vect = vect.fit(X_train)
        X_train_i = vect.transform(X_train)
        X_test_i = vect.transform(X_test)
        clf = LogisticRegression(C=C, fit_intercept=fit_intercept, max_iter=max_iter)
        clf.fit(X_train_i, y_train)
        y_predictions = clf.predict(X_test_i)
        return accuracy_score(y_test, y_predictions)


class Optimus():
    def __init__(self, params):
        self._recalculate_combinations(params)

    def _recalculate_combinations(self, params):
        self.params = params
        self.combinations = self._get_all_combinations(self.params)
        print('Optimus hyper-space is instantiated')
        print(f'Total: {len(self.combinations)} possible combinations:')
        print(self.combinations)

    def _find_prospective_difference(self, dict_1, dict_2):
        accuracy_1 = dict_1[0]
        accuracy_2 = dict_2[0]
        ls_1 = dict_1[1]
        ls_2 = dict_2[1]
        keys_1 = ls_1.keys()
        keys_2 = ls_2.keys()
        int_keys = ['max_iter', 'max_features']
        float_keys = ['C']
        prospective_params = {k: [] for k in keys_1}
        for (i, (key_1, key_2)) in enumerate(zip(keys_1, keys_2)):
            value_1 = ls_1[key_1]
            value_2 = ls_2[key_2]
            if value_1 >= value_2 and accuracy_1 >= accuracy_2 and key_1 in float_keys:
                prospective_params[key_1].append(value_1 + 0.1)
            elif value_1 <= value_2 and accuracy_1 >= accuracy_2 and key_1 in float_keys:
                prospective_params[key_1].append(value_1 - 0.1)
            elif value_1 >= value_2 and accuracy_1 >= accuracy_2 and key_1 in int_keys:
                prospective_params[key_1].append(value_1 + 5)
            elif value_1 <= value_2 and accuracy_1 >= accuracy_2 and key_1 in int_keys:
                prospective_params[key_1].append(value_1 - 5)
        return prospective_params

    def _adjust_params_from_agents(self, iteration_result):
        best_2_agents = iteration_result[0:2]
        result = self.params
        for i in best_2_agents:
            for j in best_2_agents:
                prospective_params = self._find_prospective_difference(i, j)
                for key in self.params.keys():
                    values = prospective_params[key]
                    for v in values:
                        if v not in self.params[key]:
                            self.params[key].append(v)
                            self.params[key] = self.params[key][1:]
        return self.params

    def _get_all_combinations(self, params):
        return [{'C': x, 'fit_intercept': y,
                 'ngram_range': z,
                 'max_features': i,
                 'max_iter': j,
                 'use_tf_idf': k,
                 'analyzer': l}
                for x in params['C']
                for y in params['fit_intercept']
                for z in params['ngram_range']
                for i in params['max_features']
                for j in params['max_iter']
                for k in params['use_tf_idf']
                for l in params['analyzer']]

    def _perform_hyper_search(self, train):
        X_train = train['Phrase'].values
        y_train = train['Sentiment'].values
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        result = []
        for combination in self.combinations:
            agent = TrainAgent(combination)
            accuracy = agent.get_performance(X_train, X_test, y_train, y_test)
            result.append((accuracy, combination))
            print(f'Agent is spanned with current combination: {combination}')
            print(f'Accuracy for combination is: {accuracy}')
        return sorted(result, key=lambda x: x[0], reverse=True)

    def perform_optimus_search(self, train, iter=3):
        for i in range(iter):
            print(f'Iteration: {i}')
            iteration_result = self._perform_hyper_search(train)
            new_params = self._adjust_params_from_agents(iteration_result)
            self._recalculate_combinations(new_params)
        return iteration_result


class NatualLanguage():
    def __init__(self):
        self.model = None

    def load(self, path):
        self.model = gensim.models.Word2Vec.load(path)
        return self

    def initiate(self, sg, dim=400):
        self.model = gensim.models.Word2Vec(sg, min_count=10, size=dim)
        print('Model was initiated successfully')
        return self

    def finetune(self, sg, epochs=10):
        if self.model is None:
            print('Model is not initialized')
            return
        self.model.train(sg, total_examples=len(sg), epochs=epochs)
        return self

    def save(self, bin_name):
        self.model.wv.save_word2vec_format(bin_name, binary=True)


class Planner()
    def __init__(self, simple_model, tuned_model, enhanced_model):
        self.simple_model = simple_model
        self.enhanced_model = enhanced_model
        self.tuned_model = tuned_model

    def get_combination(self, X_test):
        return self.simple_model.predict(X_test) + self.enhanced_model.predict(X_test) + self.tuned_model.predict(
            X_test)


optimus = Optimus(params)

final_result = optimus.perform_optimus_search(train_df)
