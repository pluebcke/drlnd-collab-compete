import csv
import glob
import os
import matplotlib.pyplot as plt


def create_save_path(base_path):
    #  Create directory to save the results of the next run

    prefix = "run"
    existing_dirs = glob.glob(base_path + prefix + "*")
    suffix = str(len(existing_dirs)).zfill(3)

    save_path = base_path + prefix + suffix + "/"
    try:
        os.mkdir(save_path)
        return save_path
    except OSError:
        print("Path already exists " + save_path + " no results saved")
        return


def save_results(save_path, agent, setting, rewards, average_rewards, a_loss, c_a_loss, c_b_loss):
    #  Plot some figures of the performance and save the trained neural networks

    plt.figure(1)
    plt.clf()
    plt.plot(rewards,'.')
    plt.plot(average_rewards)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt_path = save_path + "learning_curve.png"
    if os.path.isfile(plt_path):
        os.remove(plt_path)
    plt.savefig(plt_path, dpi=150)

    plt.figure(2)
    plt.clf()
    plt.plot(a_loss)
    plt.ylabel('Actor loss')
    plt.xlabel('Episode #')
    plt_path = save_path + "a_loss.png"
    if os.path.isfile(plt_path):
        os.remove(plt_path)
    plt.savefig(plt_path, dpi=150)

    plt.figure(3)
    plt.clf()
    plt.plot(c_a_loss)
    plt.plot(c_b_loss)
    plt.ylabel('Critic loss')
    plt.xlabel('Episode #')
    plt_path = save_path + "c_loss.png"
    if os.path.isfile(plt_path):
        os.remove(plt_path)
    plt.savefig(plt_path, dpi=150)

    agent.save_nets(save_path)

    file = open(save_path + 'settings.csv', 'w')
    writer = csv.DictWriter(file, fieldnames=['key', 'value'])
    writer.writeheader()
    for key in setting.keys():
        writer.writerow({'key': key, 'value': setting[key]})
    file.close()