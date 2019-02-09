# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 22:33:57 2018

@author: Meagatron
"""

from flask import Flask, render_template, request
import  SAX_sliding_half_segment

app = Flask(__name__)



@app.route('/')
def index():
    return render_template('test2.html')



@app.route('/get_data', methods=['POST','GET'])
def get_data():
    
   if request.method == 'POST':
    global y_alphabet_size 
    y_alphabet_size = request.form['y_aplha']
    global word_lenth
    word_lenth = request.form['x_aplha']
    global  window_size
    window_size = request.form['window_size']
    global  skip_offset
    skip_offset = request.form['skip_offset']
    print(SAX_sliding_half_segment.sax_visualization.get_user_data(int(y_alphabet_size),int(word_lenth),int(window_size),int(skip_offset)))
    return 'input values  %s %s %s %s <br/> <a href="/">Back hHome</a>' % (y_alphabet_size, word_lenth,window_size,skip_offset)
   else:
    y_alphabet_size=request.args.get('y_aplha')
    return ' %s have python <br/> <a href="/">Back hHome</a>' % (y_alphabet_size)

if __name__ == '__main__':
    p=SAX_sliding_half_segment.sax_visualization()
    p.distance_calculation()
    app.run()



