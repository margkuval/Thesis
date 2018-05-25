import matplotlib.pyplot as plt
import numpy as np
import datetime

def plot_best(list_iter, list_fit, list_stress, list_weight, list_defl):
    fig = plt.figure(figsize=(10, 8))

    "Fitness plot"
    list_fit = np.array(list_fit).transpose()
    x_fit = list_iter
    y_fit = list_fit
    print(list_iter)
    print(list_fit)

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(x_fit, y_fit, c='k')
    ax1.set_title('Fitness evolution')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Fitness')
    plt.grid(b=True, which='both', axis='both')

    "Stress plot"
    list_stress = np.array(list_stress).transpose()
    x_stress = list_iter
    y_stress = list_stress

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(x_stress, y_stress, c='navy')
    ax2.set_title('Stress evolution')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Abs stress sum')
    plt.grid(b=True, which='both', axis='both')

    "Weight plot"
    list_weight = np.array(list_weight).transpose()
    x_weight = list_iter
    y_weight = list_weight

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(x_weight, y_weight, c='tomato')
    ax3.set_title('Weight evolution')
    ax3.set_xlabel('Iterations')
    ax3.set_ylabel('Construction weight')
    plt.grid(b=True, which='both', axis='both')

    "Deflection plot"
    list_defl = np.array(list_defl).transpose()
    x_defl = list_iter
    print(list_defl)
    y_defl = np.round(sum(abs(list_defl[0])), 3)
    print(y_defl)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(x_defl, y_defl, c='gold')
    ax4.set_title('Deflection evolution')
    ax4.set_xlabel('Iterations')
    ax4.set_ylabel('Abs deflection sum')
    plt.grid(b=True, which='both', axis='both')

    plt.subplots_adjust(left=0.2, wspace=0.5, top=0.8, hspace=0.5)  # keep top

    plt.savefig(datetime.datetime.now().strftime('Fit_%Y%m%d_%H%M%S_') + ".pdf")
    plt.show()