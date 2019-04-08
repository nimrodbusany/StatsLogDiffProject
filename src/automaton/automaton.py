from PySimpleAutomata import DFA, NFA, automata_IO

def read_determinize_minimized_write(path, model):

    nfa = automata_IO.nfa_dot_importer(path + model)
    dfa = NFA.nfa_determinization(nfa)
    automata_IO.dfa_to_dot(dfa, model + '_det', path)
    new_dfa = DFA.dfa_minimization(dfa)
    automata_IO.dfa_to_dot(new_dfa, model + "_det_min", path)

def check_relation(nfa1, nfa2):
    '''
    :param nfa1: nfa
    :param nfa2: nfa
    :return: - 1 if model1 includes model 2
             - -1 if model2 includes model 1
             - 0 if the ktails_models are equal
             - otherwise
    '''
    dfa1 = NFA.nfa_determinization(nfa1)
    dfa1_comp = DFA.dfa_complementation(dfa1)
    dfa2 = NFA.nfa_determinization(nfa2)
    dfa2_comp = DFA.dfa_complementation(dfa2)
    words_only_in_model2 = DFA.dfa_nonemptiness_check(DFA.dfa_intersection(dfa1_comp, dfa2))
    words_only_in_model1 = DFA.dfa_nonemptiness_check(DFA.dfa_intersection(dfa2_comp, dfa1))

    if words_only_in_model1\
            and not words_only_in_model2:
        return 1
    elif words_only_in_model2\
            and not words_only_in_model1:
        dfa_intersect = DFA.dfa_intersection(dfa1_comp, dfa2)
        dfa_intersect_minimized = DFA.dfa_co_reachable(DFA.dfa_minimization(dfa_intersect))
        automata_IO.dfa_to_dot(dfa_intersect_minimized, 'csv_intersect', path)
        return -1
    elif not words_only_in_model2\
            and not words_only_in_model1:
        return 0
    else:
        return -1

def compare_models(model_path1, model_path2):

    nfa1 = automata_IO.nfa_dot_importer(model_path1)
    nfa2 = automata_IO.nfa_dot_importer(model_path2)
    print('relation:', check_relation(nfa1, nfa2))

if __name__ == '__main__':

    path = '../../data/logs/example/cvs/ktails_models/'
    compare_models(path + "cvs.net.dot", path + "csv_10__model.dot")
    compare_models(path + "cvs.net.dot", path + "csv_past_10__model.dot")
    compare_models(path + "csv_past_10__model.dot", path + "csv_10__model.dot")

    # path = '../../data/bear/ktails_models/'
    # compare_models(path + "desktop_5.dot", path + "desktop_past_5.dot")
    # compare_models(path + "mobile_5.dot", path + "mobile_past_5.dot")