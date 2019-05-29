{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "압축성유체역학.ipynb",
      "version": "0.3.2",
      "provenance": [],
      "collapsed_sections": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/socome/Compressible-flow/blob/master/%EC%95%95%EC%B6%95%EC%84%B1%EC%9C%A0%EC%B2%B4%EC%97%AD%ED%95%99.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "0LjW6COsz-yq",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "%matplotlib inline\n",
        "\n",
        "import math\n",
        "import numpy as np\n",
        "from sympy import Integral, Symbol\n",
        "from matplotlib import pyplot as plt\n",
        "from pandas import Series, DataFrame\n",
        "import pandas as pd"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "-wneUxGa0BPk",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "x_list = np.arange(0,0.5,0.001)\n",
        "a_list = np.arange(5,15,0.01)\n",
        "\n",
        "nu = 1.5*pow(10,-5)\n",
        "density = 1.225\n",
        "\n",
        "def cal_Laminar_to_Turbulence(V) :\n",
        "    \n",
        "    Laminar_to_Turbulence = 0\n",
        "    \n",
        "    for ii, x in enumerate(x_list):\n",
        "        Re = (V*x)/nu\n",
        "        if Re >= 5*pow(10,5) :\n",
        "            Laminar_to_Turbulence = x\n",
        "            break\n",
        "    return Laminar_to_Turbulence\n",
        "\n",
        "def cal_Cl_Cd(V,aoa):\n",
        "    \n",
        "    aoa_rad = aoa * math.pi / 180\n",
        "    \n",
        "    V_t = V * math.cos(aoa_rad)\n",
        "    V_n = V * math.sin(aoa_rad)\n",
        "    \n",
        "    change_x = cal_Laminar_to_Turbulence(V_t)\n",
        "    \n",
        "    x = Symbol('x')\n",
        "    f = 1/2 * density * pow(V_t,2) * 0.664 * pow(((V_t*x)/nu),-0.5)\n",
        "    laminar_friction_drag = Integral(f, (x, 0,change_x)).doit()\n",
        "    f = 1/2 * density * pow(V_t,2) * 0.0594 * pow(((V_t*x)/nu),-0.2)\n",
        "    turbulance_friction_drag = Integral(f, (x, change_x, 0.5)).doit()\n",
        "    \n",
        "    Pure_firction_Force = laminar_friction_drag + turbulance_friction_drag \n",
        "    Pure_Pressure_Force = 1/2 * density * pow(V_n,2) * 1 * 1.18\n",
        "    \n",
        "    Lift = Pure_Pressure_Force * math.sin(aoa_rad) + Pure_firction_Force * math.sin(aoa_rad)\n",
        "    Drag = Pure_Pressure_Force * math.cos(aoa_rad) - Pure_firction_Force * math.cos(aoa_rad)\n",
        "\n",
        "    Cl = Lift / (0.5 * density * pow(V,2) * 1)\n",
        "    Cd = Drag / (0.5 * density * pow(V,2) * 1)\n",
        "    \n",
        "    return Cl,Cd"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "itlTKzYA0Cic",
        "colab_type": "code",
        "outputId": "142508d3-7c6e-47cd-f2fb-c015647f5c45",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 283
        }
      },
      "source": [
        "Cl_total = []\n",
        "Cd_total = []\n",
        "\n",
        "for ii , a in enumerate(a_list): \n",
        "    Cl,Cd = cal_Cl_Cd(100,a)\n",
        "    Cl_total.append(Cl)\n",
        "    Cd_total.append(Cd)\n",
        "\n",
        "plt.plot(a_list,Cl_total)\n",
        "plt.plot(a_list,Cd_total)\n",
        "plt.xlabel('AoA')\n",
        "plt.legend(['Cl','Cd'])\n",
        "plt.show()"
      ],
      "execution_count": 25,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "bAP7zUbGGzs9",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "data = {'Cl' : Cl_total,\n",
        "       'Cd' : Cd_total}\n",
        "frame = DataFrame(data, index=a_list)\n",
        "frame.to_excel(\"Velocity_100.xlsx\")"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "82pONH570GiU",
        "colab_type": "code",
        "outputId": "3210dd17-6557-47b4-fbff-4e3c966b9b15",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 283
        }
      },
      "source": [
        "Cl_total = []\n",
        "Cd_total = []\n",
        "\n",
        "for ii , a in enumerate(a_list): \n",
        "    Cl,Cd = cal_Cl_Cd(150,a)\n",
        "    Cl_total.append(Cl)\n",
        "    Cd_total.append(Cd)\n",
        "\n",
        "plt.plot(a_list,Cl_total)\n",
        "plt.plot(a_list,Cd_total)\n",
        "plt.xlabel('AoA')\n",
        "plt.legend(['Cl','Cd'])\n",
        "plt.show()"
      ],
      "execution_count": 27,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Eu6BD6_MZ-jg",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "data = {'Cl' : Cl_total,\n",
        "       'Cd' : Cd_total}\n",
        "frame = DataFrame(data, index=a_list)\n",
        "frame.to_excel(\"Velocity_150.xlsx\")"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ve0MD7x_233u",
        "colab_type": "code",
        "outputId": "18fc2887-5214-4891-b08c-1b1d6ed06600",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 283
        }
      },
      "source": [
        "Cl_total = []\n",
        "Cd_total = []\n",
        "\n",
        "for ii , a in enumerate(a_list): \n",
        "    Cl,Cd = cal_Cl_Cd(200,a)\n",
        "    Cl_total.append(Cl)\n",
        "    Cd_total.append(Cd)\n",
        "\n",
        "plt.plot(a_list,Cl_total)\n",
        "plt.plot(a_list,Cd_total)\n",
        "plt.xlabel('AoA')\n",
        "plt.legend(['Cl','Cd'])\n",
        "plt.show()"
      ],
      "execution_count": 29,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\nxELfGGOiyP8HeB+jlU1hwjsAAAAASUVORK5CYII=\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "iPmPt72zZ_L3",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "data = {'Cl' : Cl_total,\n",
        "       'Cd' : Cd_total}\n",
        "frame = DataFrame(data, index=a_list)\n",
        "frame.to_excel(\"Velocity_200.xlsx\")"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ucTaAZ6HaESj",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        ""
      ],
      "execution_count": 0,
      "outputs": []
    }
  ]
}
# Compressible-flow HW4

[Velocity_100.xlsx](https://github.com/socome/Compressible-flow/files/3232268/Velocity_100.xlsx)

[Velocity_150.xlsx](https://github.com/socome/Compressible-flow/files/3232269/Velocity_150.xlsx)

[Velocity_200.xlsx](https://github.com/socome/Compressible-flow/files/3232270/Velocity_200.xlsx)
