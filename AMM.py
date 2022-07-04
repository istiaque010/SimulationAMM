import matplotlib
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
ranf = np.random.normal(0.0, 1.0, size=1001)

lp, ar, tr = [1000, 1000], [1000, 1000], [1000, 1000]
lp_result, ar_result, tr_result = [[1000] + [-1]*1000, [1000] + [-1]*1000], [[1000]  + [-1]*1000, [1000]  + [-1]*1000], [[1000]  + [-1]*1000, [1000]  + [-1]*1000]
time = [i for i in range(1001)]
fee = 0.01
c = [0] * 1001
f = [0] * 1001
f_sum = [0]

def init():
  lp[0], lp[1], ar[0], ar[1], tr[0], tr[1] = 1000, 1000, 1000, 1000, 1000, 1000
  f_sum[0] = 0
  for i in range(1001):
    if c[i] == 1:
      c[i] = 0
      
def update(t):
  lp_result[0][t], lp_result[1][t], ar_result[0][t], ar_result[1][t], tr_result[0][t], tr_result[1][t] = lp[0], lp[1], ar[0], ar[1], tr[0], tr[1]
  f[t] = f_sum[0]


def static_pool_price(t):
  return 1

def dynamic_pool_price(t):
  return market_price(t)

def market_price(t):
  return t/100
  
def buyorsell(pp, mp, t):
  if pp > mp:
    if ar[0] == 0:
      return 0
    move_x = ar[0]
    lp[0] += move_x
    ar[0] -= move_x

    move_y = (move_x * pp) / (1 + fee)
    lp[1] -= move_y
    ar[1] += move_y

    f_sum[0] += move_x * fee

  elif pp < mp:
    if ar[0] == 1:
      return 0
    move_y = ar[1]
    lp[1] += move_y
    ar[1] -= move_y

    move_x = (move_y / pp) / (1 + fee)
    lp[0] -= move_x
    ar[0] += move_x

    f_sum[0] += move_y * fee

def trade(pp, t):
  r = ranf[t]
  if r > 0:
    if lp[0] < r:
      c[t] += 1
    else:
      lp[0] -= r
      tr[0] += r

      lp[1] += (r * pp) * (1 + fee)
      tr[1] -= (r * pp) * (1 + fee)

      f_sum[0] += r * pp * fee

  else:
    r = -1 * r
    if lp[1] <  (r * pp) / (1 + fee):
      c[t] += 1
    else:
      lp[0] += r
      tr[0] -= r

      lp[1] -= (r * pp) / (1 + fee)
      tr[1] += (r * pp) / (1 + fee)

      f_sum[0] += r * pp / (1 + fee) * fee


init()
for t in range(1, 1001):
  pp = static_pool_price(t)
  mp = market_price(t)

  buyorsell(pp, mp, t)
  trade(pp, t)
  update(t)

v_lp, v_ar, v_tr = [], [], []
for i in range(1001):
  v_lp.append(market_price(1000) * lp_result[0][i] + lp_result[1][i])
  v_ar.append(market_price(1000) * ar_result[0][i] + ar_result[1][i])
  v_tr.append(market_price(1000) * tr_result[0][i] + tr_result[1][i])

plt.figure(figsize=(24,14))
t = [i for i in range(0, 1001, 100)]
title_font = {'fontsize': 20, 'fontweight': 'bold'}

plt.subplot(2, 3, 1)
plt.xlabel('Time')
plt.ylabel('Token X')
plt.xticks(t)
plt.yticks([i for i in range(0, 2201, 200)])
plt.plot(time, lp_result[0], time, ar_result[0], '-r', time, tr_result[0], '-g')
plt.title('(a) X token holdings', fontdict=title_font, loc='center', pad=10)

plt.subplot(2, 3, 2)
plt.xlabel('Time')
plt.ylabel('Token Y')
plt.xticks(t)
plt.yticks([i for i in range(0, 2201, 200)])
plt.plot(time, lp_result[1], time, ar_result[1], '-r', time, tr_result[1], '-g')
plt.title('(b) Y token holdings', fontdict=title_font, loc='center', pad=10)

plt.subplot(2, 3, 3)
plt.xlabel('Time')
plt.ylabel('Value')
plt.xticks(t)
plt.yticks([i for i in range(0, 20001, 2000)])
plt.plot(time, v_lp, time, v_ar, '-r', time, v_tr, '-g')
plt.title('(c) Total Value', fontdict=title_font, loc='center', pad=10)

plt.subplot(2, 3, 4)
plt.xlabel('Time')
plt.ylabel('Price')
plt.xticks(t)
plt.yticks([i for i in range(11)])
plt.plot(time, [market_price(n) for n in range(1001)], time, [static_pool_price(n) for n in range(1001)], '-r')
plt.title('(d) Market vs Pool Price', fontdict=title_font, loc='center', pad=10)

plt.subplot(2, 3, 5)
plt.xlabel('Time')
plt.ylabel('Collected Fees')
plt.xticks(t)
plt.plot(time, f)
plt.title('(e) Collected Fees', fontdict=title_font, loc='center', pad=10)

plt.subplot(2, 3, 6)
plt.xlabel('Time')
plt.ylabel('# of rejections')
plt.xticks(t)
plt.yticks([0, 1])
plt.plot(time, c, 'bo')
plt.title('(f) Declined Trades', fontdict=title_font, loc='center', pad=10)

plt.show()


init()
for t in range(1, 1001):
  pp = dynamic_pool_price(t)
  mp = market_price(t)

  buyorsell(pp, mp, t)
  trade(pp, t)
  update(t)

v_lp, v_ar, v_tr = [], [], []
for i in range(1001):
  v_lp.append(market_price(1000) * lp_result[0][i] + lp_result[1][i])
  v_ar.append(market_price(1000) * ar_result[0][i] + ar_result[1][i])
  v_tr.append(market_price(1000) * tr_result[0][i] + tr_result[1][i])
  
plt.figure(figsize=(24,14))
t = [i for i in range(0, 1001, 100)]
title_font = {'fontsize': 20, 'fontweight': 'bold'}

plt.subplot(2, 3, 1)
plt.xlabel('Time')
plt.ylabel('Token X')
plt.xticks(t)
plt.plot(time, lp_result[0], time, ar_result[0], '-r', time, tr_result[0], '-g')
plt.title('(a) X token holdings', fontdict=title_font, loc='center', pad=10)

plt.subplot(2, 3, 2)
plt.xlabel('Time')
plt.ylabel('Token Y')
plt.xticks(t)
plt.plot(time, lp_result[1], time, ar_result[1], '-r', time, tr_result[1], '-g')
plt.title('(b) Y token holdings', fontdict=title_font, loc='center', pad=10)

plt.subplot(2, 3, 3)
plt.xlabel('Time')
plt.ylabel('Value')
plt.xticks(t)
plt.plot(time, v_lp, time, v_ar, '-r', time, v_tr, '-g')
plt.title('(c) Total Value', fontdict=title_font, loc='center', pad=10)

plt.subplot(2, 3, 4)
plt.xlabel('Time')
plt.ylabel('Price')
plt.xticks(t)
plt.yticks([i for i in range(11)])
plt.plot(time, [0] + [market_price(n) for n in range(1, 1001)], time, [1] + [dynamic_pool_price(n) for n in range(1, 1001)], '-r')
plt.title('(d) Market vs Pool Price', fontdict=title_font, loc='center', pad=10)

plt.subplot(2, 3, 5)
plt.xlabel('Time')
plt.ylabel('Collected Fees')
plt.xticks(t)
plt.plot(time, f)
plt.title('(e) Collected Fees', fontdict=title_font, loc='center', pad=10)

plt.subplot(2, 3, 6)
plt.xlabel('Time')
plt.ylabel('# of rejections')
plt.xticks(t)
plt.yticks([0, 1])
plt.plot(time, c, 'bo')
plt.title('(f) Declined Trades', fontdict=title_font, loc='center', pad=10)

plt.show()
