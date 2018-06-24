import plotly.offline as pyo
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode
from IPython.display import display, HTML
import numpy as np



init_notebook_mode(connected=True)


def eigenplot(inputs):

    t=np.linspace(0, 2*np.pi, 50)
    x=np.cos(t)
    y=np.sin(t)

    mat = np.reshape(inputs, [2,2])

    U, s, vt = np.linalg.svd(mat)
    sigma = np.diag(s)

    new_x1 = vt[0,0]*x + vt[0,1]*y
    new_y1 = vt[1,0]*x + vt[1,1]*y

    new_x2 = sigma[0,0]*new_x1 + sigma[0,1]*new_y1
    new_y2 = sigma[1,0]*new_x1 + sigma[1,1]*new_y1

    new_x3 = U[0,0]*new_x2 + U[0,1]*new_y2
    new_y3 = U[1,0]*new_x2 + U[1,1]*new_y2

    lim = max(max(abs(x)), max(abs(new_x3)), max(abs(y)), max(abs(new_y3)))

    v1 = [1/np.sqrt(2), 1/np.sqrt(2)]
    v2 = [-1/np.sqrt(2), 1/np.sqrt(2)]

    new_v11 = [vt[0,0]*v1[0] + vt[0,1]*v1[1], vt[1,0]*v1[0] + vt[1,1]*v1[1]]
    new_v21 = [vt[0,0]*v2[0] + vt[0,1]*v2[1], vt[1,0]*v2[0] + vt[1,1]*v2[1]]

    new_v12 = [sigma[0,0]*new_v11[0] + sigma[0,1]*new_v11[1], sigma[1,0]*new_v11[0] + sigma[1,1]*new_v11[1]]
    new_v22 = [sigma[0,0]*new_v21[0] + sigma[0,1]*new_v21[1], sigma[1,0]*new_v21[0] + sigma[1,1]*new_v21[1]]


    new_v13 = [U[0,0]*new_v12[0] + U[0,1]*new_v12[1], U[1,0]*new_v12[0] + U[1,1]*new_v12[1]]
    new_v23 = [U[0,0]*new_v22[0] + U[0,1]*new_v22[1], U[1,0]*new_v22[0] + U[1,1]*new_v22[1]]

    data = [go.Scatter(x=x,
                       y=y,
                       mode='markers',
                       name='unit circle',
                       marker=dict(size=5,
                                   color='black')
                      ),

            go.Scatter(x=[0,v1[0]],
                       y=[0,v1[1]],
                       mode='lines',
                       name='first eigenvector',
                       line=dict(width=3,
                                 color='blue')
                      ),

            go.Scatter(x=[0,v2[0]],
                       y=[0,v2[1]],
                       name='second eigenvector',
                       mode='lines',
                       line=dict(width=3,
                                 color='red')
                      )
           ]

    frames=[dict(data=[go.Scatter(x=new_x1,
                                  y=new_y1,
                                  mode='markers',
                                  marker=dict(size=10,
                                              color='red')
                                 ),

                       go.Scatter(x=[0,new_v11[0]],
                                  y=[0,new_v11[1]],
                                  mode='lines',
                                  name='eigenvector 1',
                                  line=dict(width=3,
                                            color='blue')
                                 ),

                       go.Scatter(x=[0,new_v21[0]],
                                  y=[0,new_v21[1]],
                                  mode='lines',
                                  name='eigenvector 2',
                                  line=dict(width=3,
                                            color='red')
                                 )
                      ]
                ),

            dict(data=[go.Scatter(x=new_x2,
                                  y=new_y2,
                                  mode='markers',
                                  marker=dict(size=10,
                                              color='red')
                                 ),

                       go.Scatter(x=[0,new_v12[0]],
                                  y=[0,new_v12[1]],
                                  mode='lines',
                                  name='eigenvector 1',
                                  line=dict(width=3,
                                            color='blue')
                                 ),

                       go.Scatter(x=[0,new_v22[0]],
                                  y=[0,new_v22[1]],
                                  mode='lines',
                                  name='eigenvector 2',
                                  line=dict(width=3,
                                            color='red')
                                 )
                      ]
                ),

            dict(data=[go.Scatter(x=new_x3,
                                  y=new_y3,
                                  mode='markers',
                                  marker=dict(size=10,
                                              color='red')
                                 ),

                       go.Scatter(x=[0,new_v13[0]],
                                  y=[0,new_v13[1]],
                                  mode='lines',
                                  name='eigenvector 1',
                                  line=dict(width=3,
                                            color='blue')
                                 ),

                       go.Scatter(x=[0,new_v23[0]],
                                  y=[0,new_v23[1]],
                                  mode='lines',
                                  name='eigenvector 2',
                                  line=dict(width=3,
                                            color='red')
                                 )
                      ]
                ),



            dict(data=[go.Scatter(x=x,
                                  y=y,
                                  mode='markers',
                                  name='unit circle',
                                  marker=dict(size=5,
                                              color='black')
                                 ),

                       go.Scatter(x=[0,v1[0]],
                                  y=[0,v1[1]],
                                  mode='lines',
                                  name='eigenvector 1',
                                  line=dict(width=3,
                                            color='blue')
                                 ),

                       go.Scatter(x=[0,v2[0]],
                                  y=[0,v2[1]],
                                  name='eigenvector 2',
                                  mode='lines',
                                  line=dict(width=3,
                                            color='red')
                                 )
                      ]
                )
           ]

    layout = go.Layout(xaxis=dict(range=[-lim, lim],
                                  autorange=False,
                                  zeroline=False),

                       yaxis=dict(range=[-lim, lim],
                                  autorange=False,
                                  zeroline=False),

                       title='Linear transformation of plane', hovermode='closest',

                       updatemenus=[{'type': 'buttons',
                                     'buttons': [{'label': 'Transform',
                                                  'method': 'animate',
                                                  'args': [None]}
                                                ]
                                    }]
                      )

    figure = go.Figure(data=data, layout=layout, frames=frames)

    return figure

pyo.plot(eigenplot([1,2, 2, 1]), filename='plot3.html')
