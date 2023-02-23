import matplotlib.pyplot as plt
import numpy as np

def main():
    data = np.load('cleaned_data.npy')
    plot_colors = ['k', 'm', 'g', 'y', 'r', 'b', 'c']
    cnames = ['Purple', 'Pink', 'Dark Green', 'Yellow', 'Red', 'Blue', 'Light Green']

    for color in range(len(data[0])):
        fig, axs = plt.subplots(2, sharex=True)
        fig.suptitle("Data for " +cnames[color])
        axs[0].plot(data[:-10,color,0], plot_colors[color])
        axs[0].set_title("Radius")
        axs[1].plot(data[:-10,color,1], plot_colors[color])
        axs[1].set_title("Angle")
    plt.show()


if __name__ == '__main__':
    main()
