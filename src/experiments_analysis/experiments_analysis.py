import pandas as pd
import matplotlib.pyplot as plt
from src.utils.disk_operations import create_folder_if_missing

def plot_box(data, grp_bys, column_, x_label_="", y_label_="", title_="", \
             output_path=None, showfliers_=False):


    ax = data.boxplot(column=column_, by= grp_bys, showfliers=showfliers_)
    if x_label_:
        ax.set_xlabel(x_label_)
    if y_label_:
        ax.set_ylabel(y_label_)
    if title_:
        ax.set_title(title_)
    if output_path:
        plt.savefig(output_path)
        plt.clf()
        plt.cla()
    else:
        plt.show()


def make_name_consistent_with_filesystem(name):
    return name.replace("(", "[").replace(")", "]").replace(" ", "_")


def create_boxplot_per_projection(data, grp_bys, column_, x_label_, y_label, project_by_=[], output_path=None, show_outliers=False):
    '''
    creates a set of box plot, one pre projection; projection is done by the project_by_ columns
    :param data:
    :param grp_bys:
    :param column_:
    :param x_label_:
    :param y_label:
    :param project_by_:
    :param output_path:
    :return:
    '''
    for project_by_column in project_by_:
        for v in set(data[project_by_column]):
            grp_label = project_by_column + "_" + str(v)
            projected_data = data[data[project_by_column]==v]
            output_ = output_path + "fig_" + grp_label
            plot_box(projected_data, grp_bys, column_, x_label_, y_label, grp_label, output_, show_outliers)

def print_data_summary(data, by_):
    print("========", 'VARYING', str(by_).upper(), 'EXPERIMENT SUMMARY', "========")
    print("alpha", round(data['true_error (tp/tn+fp)'].mean(), 2)) ## , round(data['statistical_error_clean'].mean(), 2)
    print("power", round(data['power (tp/tp+fn)'].mean(), 2))
    # grps = data.groupby(['model', by_])
    # print(grps[['tp', 'fp', 'tn', 'fn']].mean())
    grps = data.groupby([by_])
    print(grps[['true_error (tp/tn+fp)', 'power (tp/tp+fn)']].mean())
def create_boxplot_for_metric(data, attribute, label, output_folder, tup, filter_nones=True, secondary_grps=None, show_outliers=True):

    data_alpha_defined = data[data[attribute] != -1.0] if filter_nones else data
    plot_box(data_alpha_defined, grp_bys=tup[0], column_=attribute, x_label_=tup[1], \
             y_label_=label, output_path=output_folder + '/' + tup[1] + '_' + label, showfliers_=show_outliers)
    if secondary_grps:
        groups = data.groupby(secondary_grps)
        for grp, grp_df in groups:
            plot_box(grp_df, grp_bys=tup[0], column_=attribute, x_label_=tup[1], \
                     y_label_=label, title_=grp, output_path=tup[2] + grp + '_' + label, showfliers_=show_outliers)


def create_algorithms_kdiff_plots(exp_id, algorithm, results_fname='results.csv', show_outliers=False):

    if algorithm not in ['s2kdiff', 'snkdiff']:
        raise ValueError("Bad value")

    BASE_DIR = "../../results/statistical_experiments/paper_experiments/model_based/"+ algorithm + "/" + exp_id + "/"
    OUTPUT_DIR = "../../results/statistical_experiments/paper_experiments/analysis/"+ algorithm + "/" + exp_id + "/"
    create_folder_if_missing(OUTPUT_DIR)
    columns2keep_ = ['model', 'log_size', 'k', 'alpha', 'min_diff', \
                     'true_error (tp/tn+fp)', 'power (tp/tp+fn)', 'num_of_logs', \
                   'tp','fp', 'tn', 'fn'] ## , 'stat_tp','stat_fp', 'stat_tn', 'stat_fn', 'statistical_error_clean',
    column_alpha, column_beta, y_label_alpha, y_label_beta = 'true_error (tp/tn+fp)', 'power (tp/tp+fn)' \
        , 'error rate (alpha)', 'power (beta)',

    exps = [
            # (['k'], 'k', OUTPUT_DIR + "/k/", BASE_DIR + "varying_k/"), \
            # (['alpha'], 'alpha', OUTPUT_DIR + "/alpha/" , BASE_DIR + "varying_alpha/"),  \
            (['log_size'], 'log_size', OUTPUT_DIR + "/log_size/",  BASE_DIR + "varying_sample_size/"), \
            (['min_diff'], 'diff', OUTPUT_DIR + "/diff/",  BASE_DIR + "varying_min_diff/"),
            # (['num_of_logs'], 'num_of_logs', OUTPUT_DIR + "/num_of_logs/", BASE_DIR + "varying_num_of_logs/")
        ]

    secondary_grps = ["model"]
    for tup in exps:
        create_folder_if_missing(tup[2])
        data = read_result_file(tup[3] + results_fname, columns2keep_)
        print_data_summary(data, tup[0][0])
        # create_boxplot_for_metric(data, 'true_error (tp/tn+fp)', 'alpha', OUTPUT_DIR, tup, secondary_grps=secondary_grps, show_outliers=show_outliers)
        # create_boxplot_for_metric(data, 'power (tp/tp+fn)', 'beta', OUTPUT_DIR, tup, secondary_grps=secondary_grps, show_outliers=show_outliers)
        # plot_box(data_alpha_defined, grp_bys=tup[0], column_='statistical_error_clean', x_label_=tup[1], y_label_='statistical_error_clean',
        #          output_path=tup[2] + 'alpha_clean' + PNG_SUFFIX)



def read_result_file(results_file, columns2keep=None):
    data = pd.read_csv(results_file)
    if columns2keep:
        data = data[columns2keep]
    return data


if __name__ == '__main__':

    # exp_id = "exp_id_xHtQjyDM7b4OLuIz"
    # exp_id = "exp_id_SLOD0WhTuCVmt5xA"
    exp_id = "exp_id_MJesi2H4KfX71lSZ"
    exp_id = "exp_id_4TNGQcP6DCfVQ7TK"
    exp_id = "exp_id_xHtQjyDM7b4OLuIz"

    # exp_id_xHtQjyDM7b4OLuIz
    create_algorithms_kdiff_plots(exp_id, 's2kdiff', 'results_z_tests.csv', show_outliers=True)
    # print('====nkdiff====')
    # create_algorithms_kdiff_plots(exp_id, 'snkdiff', 'results_z_tests.csv', show_outliers=True)
    # create_algorithms_kdiff_plots(exp_id, 'snkdiff', 'results_chi_square.csv', show_outliers=True)
