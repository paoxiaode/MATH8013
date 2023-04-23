import numpy as np


def getWini(
    hidden_units=[10, 20, 40],
    input_dim=1,
    output_dim_final=1,
    astddev=0.05,
    bstddev=0.05,
):
    hidden_num = len(hidden_units)
    # print(hidden_num)
    add_hidden = [input_dim] + hidden_units

    w_Univ0 = []
    b_Univ0 = []

    for i in range(hidden_num):
        input_dim = add_hidden[i]
        output_dim = add_hidden[i + 1]
        ua_w = np.float32(
            np.random.normal(loc=0.0, scale=astddev, size=[input_dim, output_dim])
        )
        ua_b = np.float32(np.random.normal(loc=0.0, scale=bstddev, size=[output_dim]))
        w_Univ0.append(np.transpose(ua_w))
        b_Univ0.append(np.transpose(ua_b))
    ua_w = np.float32(
        np.random.normal(
            loc=0.0,
            scale=astddev,
            size=[hidden_units[hidden_num - 1], output_dim_final],
        )
    )
    ua_b = np.float32(np.random.normal(loc=0.0, scale=bstddev, size=[output_dim_final]))
    w_Univ0.append(np.transpose(ua_w))
    b_Univ0.append(np.transpose(ua_b))
    return w_Univ0, b_Univ0


def my_fft(
    data, freq_len=40, x_input=np.zeros(10), kk=0, min_f=0, max_f=np.pi / 3, isnorm=1
):
    second_diff_input = np.mean(np.diff(np.diff(np.squeeze(x_input))))
    if abs(second_diff_input) < 1e-10:
        datat = np.squeeze(data)
        datat_fft = np.fft.fft(datat)
        freq_len = min(freq_len, len(datat_fft))
        # print(freq_len)
        ind2 = range(freq_len)
        fft_coe = datat_fft[ind2]
        if isnorm == 1:
            return_fft = np.absolute(fft_coe)
        else:
            return_fft = fft_coe
    else:
        return_fft = get_ft_multi(
            x_input,
            data,
            kk=kk,
            freq_len=freq_len,
            min_f=min_f,
            max_f=max_f,
            isnorm=isnorm,
        )
    return return_fft


def get_ft_multi(x_input, data, kk=0, freq_len=100, min_f=0, max_f=np.pi / 3, isnorm=1):
    n = x_input.shape[1]
    if np.max(abs(kk)) == 0:
        k = np.linspace(min_f, max_f, num=freq_len, endpoint=True)
        kk = np.matmul(np.ones([n, 1]), np.reshape(k, [1, -1]))
    tmp = np.matmul(np.transpose(data), np.exp(-1j * (np.matmul(x_input, kk))))
    if isnorm == 1:
        return_fft = np.absolute(tmp)
    else:
        return_fft = tmp
    return np.squeeze(return_fft)


def SelectPeakIndex(FFT_Data, endpoint=True):
    D1 = FFT_Data[1:-1] - FFT_Data[0:-2]
    D2 = FFT_Data[1:-1] - FFT_Data[2:]
    D3 = np.logical_and(D1 > 0, D2 > 0)
    tmp = np.where(D3 == True)
    sel_ind = tmp[0] + 1
    if endpoint:
        if FFT_Data[0] - FFT_Data[1] > 0:
            sel_ind = np.concatenate([[0], sel_ind])
        if FFT_Data[-1] - FFT_Data[-2] > 0:
            Last_ind = len(FFT_Data) - 1
            sel_ind = np.concatenate([sel_ind, [Last_ind]])
    return sel_ind
