from cProfile import label
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

def plot3(X, y, model, ith_scan):
    line_width = 0.5 # For plotting
    q_resolution = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    cell_max_min = (x_min, x_max, y_min, y_max)
    x = np.arange(cell_max_min[0], cell_max_min[1]+1, q_resolution)
    y = np.arange(cell_max_min[2], cell_max_min[3]+1, q_resolution)
    xx, yy = np.meshgrid(x,y)
    X_query = np.hstack((xx.ravel()[:, np.newaxis], yy.ravel()[:, np.newaxis]))
    Y_query = model.predict(X_query)
    cs1 = plt.contour(x, y, np.reshape(Y_query, (len(y), len(x))), cmap="tab20c", linestyles="solid", linewidths=line_width)
    plt.xlim([cell_max_min[0], cell_max_min[1]])
    plt.ylim([cell_max_min[2], cell_max_min[3]])
    pos_points = model.pos_points
    neg_points = model.neg_points
    plt.scatter(pos_points[:, 0], pos_points[:,1], color='red', s=5, edgecolors=['none'],label='pos sp')
    plt.scatter(neg_points[:, 0], neg_points[:,1], color='blue', s=5, edgecolors=['none'],label='neg sp')
    plt.legend(loc='upper right')
    plt.savefig('./outputs/imgs/gamma_{}_scan_{}.png'.format(model.gamma ,ith_scan))
    plt.close()
    # plt.show()

def plot5(X, y, model, ith_scan):
    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    cell_max_min = (x_min, x_max, y_min, y_max)
    # cell_max_min = (-20, 20, -25, 10)
    plt.xlim([cell_max_min[0], cell_max_min[1]])
    plt.ylim([cell_max_min[2], cell_max_min[3]])
    pos_ids = np.nonzero(y+1)
    neg_ids = np.nonzero(y-1)
    pos_points = X[pos_ids[0]]
    neg_points = X[neg_ids[0]]
    plt.scatter(pos_points[:, 0], pos_points[:,1], color='red', s=5, edgecolors=['none'],label='pos sp')
    plt.scatter(neg_points[:, 0], neg_points[:,1], color='blue', s=5, edgecolors=['none'],label='neg sp')
    plt.legend(loc='upper right')
    plt.title('Data points at t={}'.format(ith_scan))

    plt.subplot(122)
    plt.xlim([cell_max_min[0], cell_max_min[1]])
    plt.ylim([cell_max_min[2], cell_max_min[3]])
    # pos_points = model.whole_pos_points
    # neg_points = model.whole_neg_points
    pos_points = model.pos_points
    neg_points = model.neg_points
    plt.scatter(pos_points[:, 0], pos_points[:,1], color='red', s=5, edgecolors=['none'],label='pos sp')
    plt.scatter(neg_points[:, 0], neg_points[:,1], color='blue', s=5, edgecolors=['none'],label='neg sp')
    plt.legend(loc='upper right')
    plt.title('support points at t={}'.format(ith_scan))
    plt.savefig('./outputs/imgs/concat_scan_{}.png'.format(ith_scan))
    # plt.show()
    # plt.close("all")

def debug(model):
    print("amount of support points: ", model.numberSupportPoints)
    print("alpha: ", model.alpha)
    print("+ alpha: ", model.pos_alpha)
    print("- alpha: ", model.neg_alpha)
    print("dicesion boundary: ", model.F)
    print("support point: ",model.data)
    print("+ support point: ",model.pos_points)
    print("- support point: ",model.neg_points)
    print(model.G)

def save_params(data):
    df = pd.DataFrame(data, columns=['ith_scan','data_points','iterations','time cost'])
    print(df)
    df.to_csv('./outputs/params/log_origin.csv')
    # df.to_csv('./outputs/params/log_onestep.csv')


def toVideo():
    img_root = './outputs/global/'  # 是图片序列的位置
    fps = 20  # 可以随意调整视频的帧速率

    #可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoWriter = cv2.VideoWriter('./outputs/video/TestVideo.avi',fourcc,fps,(2000,1000),True)#最后一个是保存图片的尺寸

    for i in range(200):
        frame = cv2.imread(img_root+'scan_{}.png'.format(i))
        cv2.imshow('frame',frame)
        videoWriter.write(frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    videoWriter.release()
    cv2.destroyAllWindows()


