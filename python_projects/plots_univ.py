import matplotlib.pyplot as plt
import numpy as np
import datetime


def plot_best_1(list_iter, list_fit, list_stress_positive, list_stress_negative, list_weight, list_defl):
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle('Individual 1: the fittest individual development in time',
                 horizontalalignment='center', verticalalignment='center')

    "Fitness plot"
    list_fit = np.array(list_fit).transpose()
    x_fit = list_iter
    y_fit = list_fit

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(x_fit, y_fit, c='r')
    ax1.set_title('Fitness evolution')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Fitness')
    plt.grid(b=True, which='both', axis='both')

    "Stress plot"
    list_stress_positive = np.array(list_stress_positive)
    x_stress_positive = list_iter
    y_stress_positive = list_stress_positive
    stress_positive = (x_stress_positive, y_stress_positive)

    list_stress_negative = np.array(list_stress_negative)
    x_stress_negative = list_iter
    y_stress_negative = list_stress_negative

    stress_negative = (x_stress_negative, y_stress_negative)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title('Stress evolution')
    ax2.set_xlabel('Iterations')
    ax2.plot(x_stress_negative, y_stress_negative, c='coral')
    ax2.set_ylabel('Negative stress sum (MPa)')
    ax2.yaxis.label.set_color('coral')

    ax2_t = ax2.twinx()
    ax2_t.plot(x_stress_positive, y_stress_positive, c='darkslateblue')
    ax2_t.set_ylabel('Positive stress sum (MPa)')
    plt.grid(b=True, which='both', axis='both')
    ax2_t.yaxis.label.set_color('darkslateblue')

    "Weight plot"
    list_weight = np.array(list_weight).transpose()
    x_weight = list_iter
    y_weight = list_weight

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(x_weight, y_weight, c='firebrick')
    ax3.set_title('Weight evolution')
    ax3.set_xlabel('Iterations')
    ax3.set_ylabel('Construction weight (1000kg)')
    plt.grid(b=True, which='both', axis='both')

    "Deflection plot"
    list_defl = np.array(list_defl).transpose()
    x_defl = list_iter
    y_defl = np.round(sum(abs(list_defl[0])), 3)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(x_defl, y_defl, c='lightcoral')
    ax4.set_title('Deflection evolution')
    ax4.set_xlabel('Iterations')
    ax4.set_ylabel('Abs deflection sum (m)')
    plt.grid(b=True, which='both', axis='both')

   # plt.legend(bbox_to_anchor=(0., 1.007, 1., .101), loc=3, ncol=1, mode="expand", borderaxespad=0.)

    plt.subplots_adjust(wspace=0.5, hspace=0.5)  # keep top
    plt.savefig(datetime.datetime.now().strftime('F_snp_w_d_I1_%Y%m%d_%H%M%S_') + ".pdf")


def plot_best_2(list_iter_2, list_fit_2, list_stress_positive_2, list_stress_negative_2, list_weight_2, list_defl_2):
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle('Individual 2: the fittest individual development in time',
                 horizontalalignment='center', verticalalignment='center')

    "Fitness plot"
    list_fit_2 = np.array(list_fit_2).transpose()
    x_fit = list_iter_2
    y_fit = list_fit_2

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(x_fit, y_fit, c='navy')
    ax1.set_title('Fitness evolution')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Fitness')
    plt.grid(b=True, which='both', axis='both')

    "Stress plot"
    list_stress_positive_2 = np.array(list_stress_positive_2)
    x_stress_positive_2 = list_iter_2
    y_stress_positive_2 = list_stress_positive_2
    stress_positive = (x_stress_positive_2, y_stress_positive_2)

    x_stress_negative_2 = list_iter_2
    y_stress_negative_2 = np.array(list_stress_negative_2)
    stress_negative = (x_stress_negative_2, y_stress_negative_2)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title('Stress evolution')
    ax2.set_xlabel('Iterations')
    ax2.plot(x_stress_negative_2, y_stress_negative_2, c='coral')
    ax2.set_ylabel('Negative stress sum (MPa)')
    ax2.yaxis.label.set_color('coral')

    ax2_t = ax2.twinx()
    ax2_t.plot(x_stress_positive_2, y_stress_positive_2, c='darkslateblue')
    ax2_t.set_ylabel('Positive stress sum (MPa)')
    plt.grid(b=True, which='both', axis='both')
    ax2_t.yaxis.label.set_color('darkslateblue')

    "Weight plot"
    list_weight = np.array(list_weight_2).transpose()
    x_weight = list_iter_2
    y_weight = list_weight

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(x_weight, y_weight, c='cornflowerblue')
    ax3.set_title('Weight evolution')
    ax3.set_xlabel('Iterations')
    ax3.set_ylabel('Construction weight (1000kg)')
    plt.grid(b=True, which='both', axis='both')

    "Deflection plot"
    list_defl = np.array(list_defl_2).transpose()
    x_defl = list_iter_2
    y_defl = np.round(sum(abs(list_defl[0])), 3)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(x_defl, y_defl, c='mediumblue')
    ax4.set_title('Deflection evolution')
    ax4.set_xlabel('Iterations')
    ax4.set_ylabel('Abs deflection sum (m)')
    plt.grid(b=True, which='both', axis='both')

   # plt.legend(bbox_to_anchor=(0., 1.007, 1., .101), loc=3, ncol=1, mode="expand", borderaxespad=0.)

    plt.subplots_adjust(wspace=0.5, hspace=0.5)  # keep top
    plt.savefig(datetime.datetime.now().strftime('F_snp_w_d_I2_%Y%m%d_%H%M%S_') + ".pdf")


def plot_best_3(list_iter_3, list_fit_3, list_stress_positive_3, list_stress_negative_3, list_weight_3, list_defl_3):
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle('Individual 3: the fittest individual development in time',
                 horizontalalignment='center', verticalalignment='center')

    "Fitness plot"
    list_fit_3 = np.array(list_fit_3).transpose()
    x_fit = list_iter_3
    y_fit = list_fit_3

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(x_fit, y_fit, c='gold')
    ax1.set_title('Fitness evolution')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Fitness')
    plt.grid(b=True, which='both', axis='both')

    "Stress plot"
    list_stress_positive = np.array(list_stress_positive_3)
    x_stress_positive = list_iter_3
    y_stress_positive = list_stress_positive
    stress_positive = (x_stress_positive, y_stress_positive)

    list_stress_negative = np.array(list_stress_negative_3)
    x_stress_negative = list_iter_3
    y_stress_negative = list_stress_negative
    stress_negative = (x_stress_negative, y_stress_negative)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title('Stress evolution')
    ax2.set_xlabel('Iterations')
    ax2.plot(x_stress_negative, y_stress_negative, c='coral')
    ax2.set_ylabel('Negative stress sum (MPa)')
    ax2.yaxis.label.set_color('coral')

    ax2_t = ax2.twinx()
    ax2_t.plot(x_stress_positive, y_stress_positive, c='darkslateblue')
    ax2_t.set_ylabel('Positive stress sum (MPa)')
    plt.grid(b=True, which='both', axis='both')
    ax2_t.yaxis.label.set_color('darkslateblue')

    "Weight plot"
    list_weight = np.array(list_weight_3).transpose()
    x_weight = list_iter_3
    y_weight = list_weight

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(x_weight, y_weight, c='goldenrod')
    ax3.set_title('Weight evolution')
    ax3.set_xlabel('Iterations')
    ax3.set_ylabel('Construction weight (1000kg)')
    plt.grid(b=True, which='both', axis='both')

    "Deflection plot"
    list_defl = np.array(list_defl_3).transpose()
    x_defl = list_iter_3
    y_defl = np.round(sum(abs(list_defl[0])), 3)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(x_defl, y_defl, c='y')
    ax4.set_title('Deflection evolution')
    ax4.set_xlabel('Iterations')
    ax4.set_ylabel('Abs deflection sum (m)')
    plt.grid(b=True, which='both', axis='both')

   # plt.legend(bbox_to_anchor=(0., 1.007, 1., .101), loc=3, ncol=1, mode="expand", borderaxespad=0.)

    plt.subplots_adjust(wspace=0.5, hspace=0.5)  # keep top
    plt.savefig(datetime.datetime.now().strftime('F_snp_w_d_I3_%Y%m%d_%H%M%S_') + ".pdf")


"""
def plot_best_1_alter(list_iter, list_fit, list_stress, list_weight, list_defl):
    fig = plt.figure(figsize=(10, 8))

    fig.suptitle('Individual 1: the fittest individual development in time',
                 horizontalalignment='center', verticalalignment='center')

    "Positive stress plot"
    list_fit = np.array(list_fit).transpose()
    x_fit = list_iter
    y_fit = list_fit

    list_iter = []
    list_fit = []
    list_weight = []
    list_stress = []
    list_stress_positive = []
    list_stress_negative = []
    list_defl = []

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(x_fit, y_fit, c='orange')
    ax1.set_title('Positive stress evolution')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Positive stress sum (MPa)')
    plt.grid(b=True, which='both', axis='both')

    "Negative stress plot"
    list_stress = np.array(list_stress).transpose()
    x_stress = list_iter
    y_stress = list_stress

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(x_stress, y_stress, c='coral')
    ax2.set_title('Negative stress evolution')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Negative stress sum (MPa)')
    plt.grid(b=True, which='both', axis='both')

    "Weight plot"
    list_weight = np.array(list_weight).transpose()
    x_weight = list_iter
    y_weight = list_weight

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(x_weight, y_weight, c='firebrick')
    ax3.set_title('Weight evolution')
    ax3.set_xlabel('Iterations')
    ax3.set_ylabel('Construction weight (1000kg)')
    plt.grid(b=True, which='both', axis='both')

    "Deflection plot"
    list_defl = np.array(list_defl).transpose()
    x_defl = list_iter
    y_defl = np.round(sum(abs(list_defl[0])), 3)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(x_defl, y_defl, c='lightcoral')
    ax4.set_title('Deflection evolution')
    ax4.set_xlabel('Iterations')
    ax4.set_ylabel('Abs deflection sum (m)')
    plt.grid(b=True, which='both', axis='both')

    plt.legend(bbox_to_anchor=(0., 1.007, 1., .101), loc=3,
               ncol=1, mode="expand", borderaxespad=0.)

    plt.subplots_adjust(wspace=0.5, top=0.8, hspace=0.5)  # keep top

    plt.savefig(datetime.datetime.now().strftime('F_s_w_d_1x4_%Y%m%d_%H%M%S_') + ".pdf")"""


"""def plot_fits(list_iter, list_iter_2, list_fit, list_fit_2):
    list_fit = np.array(list_fit).transpose()

    "Fitness plot"
    x_fit_1 = list_iter
    x_fit_2 = list_iter_2
    y_fit_1 = list_fit
    y_fit_2 = list_fit_2
    print(x_fit_1)
    print(y_fit_1)
    print(x_fit_2)
    print(y_fit_2)

    #plt.plot(x_fit_1, y_fit_1)
    #plt.plot(x_fit_2, y_fit_2)
    #plt.show()

    fig = plt.figure(figsize=(10, 8))

    "Fitness plot"

    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(x_fit_1, y_fit_1, c='k')
    ax1.set_title('Fitness evolution')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Fitness')
    plt.grid(b=True, which='both', axis='both')

    ax2 = fig.add_subplot(1, 1, 1, sharex=ax1, sharey=ax1)
    ax2.plot(x_fit_2, y_fit_2, c='navy')
    plt.grid(b=True, which='both', axis='both')

    plt.subplots_adjust(left=0.2, wspace=0.5, top=0.8, hspace=0.5)  # keep top

    plt.savefig(datetime.datetime.now().strftime('Fit_%Y%m%d_%H%M%S_') + ".pdf")"""


def plot_fits_3(list_iter, list_iter_2, list_iter_3,
                list_fit, list_fit_2, list_fit_3,
                population_1, population_2, population_3,
                mut_x_1, mut_x_2, mut_x_3, mut_yA_1, mut_yA_2, mut_yA_3):

    list_fit = np.array(list_fit).transpose()
    list_fit_2 = np.array(list_fit_2).transpose()
    list_fit_3 = np.array(list_fit_3).transpose()

    fig = plt.figure(figsize=(10, 8))

    fig.suptitle('Fitness evolution', horizontalalignment='center', verticalalignment='center')
    fig.subplots_adjust(wspace=0.5, hspace=0.6)

    "Fitness plot"
    x_fit_1 = list_iter
    y_fit_1 = list_fit
    x_fit_2 = list_iter_2
    y_fit_2 = list_fit_2
    x_fit_3 = list_iter_3
    y_fit_3 = list_fit_3

    ax1 = fig.add_subplot(1, 1, 1)

    ax1.plot(x_fit_1, y_fit_1, 'r', label=('Population size: %s' % population_1,
                                           'Mutation x: every %s' % mut_x_1, 'Mutation y, A: every %s' %mut_yA_1))
    ax1.plot(x_fit_2, y_fit_2, 'navy',label=('Population size: %s' % population_2,
                                             'Mutation x: every %s' % mut_x_2, 'Mutation y, A: every %s' %mut_yA_2))
    ax1.plot(x_fit_3, y_fit_3, 'gold', label=('Population size: %s' % population_3,
                                              'Mutation x: every %s' % mut_x_3, 'Mutation y, A: every %s' %mut_yA_3))

    #ax1.plot(x_fit_1, y_fit_1, 'r', label='Mutation %s' % )
    #ax1.plot(x_fit_2, y_fit_2, 'navy',label='Pop %s' % population_2)
    #ax1.plot(x_fit_3, y_fit_3, 'gold', label='Pop %s' % population_3)
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Fitness')
    plt.grid(b=True, which='both', axis='both')
    plt.legend(bbox_to_anchor=(0., 1.007, 1., .101), loc=3,
               ncol=1, mode="expand", borderaxespad=0.)

    plt.savefig(datetime.datetime.now().strftime('Fit3_%Y%m%d_%H%M%S_') + ".pdf")
