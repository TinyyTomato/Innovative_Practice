
// 柱形图模块 使用立即执行函数目的为了方式变量污染
(function() {
  // 1. 实例化对象
  var myChart = echarts.init(document.querySelector(".bar .chart"));
  // 2. 指定配置选项 和数据
  var option = {
    //   修改柱子颜色
    color: ["#2f89cf"],
    tooltip: {
      trigger: "axis",
      axisPointer: {
        // 坐标轴指示器，坐标轴触发有效
        type: "shadow" // 默认为直线，可选为：'line' | 'shadow'
      }
    },
    // grid决定我们的柱状图的大小
    grid: {
      left: "0%",
      top: "10px",
      right: "0%",
      bottom: "4%",
      containLabel: true
    },
    xAxis: [
      {
        type: "category",
        data: [
          "中国",
          "美国",
          "英国",
          "印度",
          "法国",
          "德国",
          "俄罗斯"
        ],
        axisTick: {
          alignWithLabel: true
        },
        // x轴的文字颜色和大小
        axisLabel: {
          color: "#5677fc",
          fontSize: "11"
        },
        // 不显示x轴的线
        axisLine: {
          show: false
        }
      }
    ],
    yAxis: [
      {
        type: "value",
        // y轴的文字颜色和大小
        axisLabel: {
          color: "#5677fc",
          fontSize: "12"
        },
        // y轴分割线的颜色
        splitLine: {
          lineStyle: {
            color: "#5677fc"
          }
        }
      }
    ],
    series: [
      {
        name: "新增人数",
        type: "bar",
        // 柱子的宽度
        barWidth: "30%",
        data: [401, 70537, 5604, 14250, 65375, 21974, 9221],
        itemStyle: {
          // 修改柱子圆角
          barBorderRadius: 3
        }
      }
    ]
  };

  // 3. 把配置选项给 实例对象
  myChart.setOption(option);
  // 让我们的图表适配屏幕宽度
  window.addEventListener("resize", function() {
    //   让我们图表调用resize方法
    myChart.resize();
  });
})();
// map地图航线模块
(function() {
  var myChart = echarts.init(document.querySelector(".map .chart"));
  var option;

  option ={
    backgroundColor: '#013954',
    dataset:{
      "source": [
        ["湖南", "112.929238", "28.249957", "90", "28", "66", "rgba(19,198,249,1)"],
        ["上海", "121.481115", "31.234694", "88", "48", "75", "rgba(19,198,249,1)"],
        ["湖北", "114.311073", "30.601099", "79", "56", "85", "rgba(245, 42, 118,1)"],
        ["澳门", "113.545265", "22.170938", "82", "36", "68", "rgba(245, 42, 118,1)"],
        ["江苏", "119.750934", "33.196174", "82", "36", "68", "rgba(245, 42, 118,1)"]
      ]
    },
    "tooltip": {
      "triggerOn": "mousemove",
      "backgroundColor": "transparent",
      "alwaysShowContent": true,
      // "formatter": "function formatter(params) {\n            var html = \"<div style=\\\"width:140px;height:80px;background-image:url(/UploadFile/ImgSource/ZH092468/a42d82bce9344df1825d70ef0210b9e4.png);background-size:100% 100%;background-repeat:no-repeat;background-position:center;padding-top:5px;padding-left:10px;box-sizing:border-box;\\\">\\n                      <div style=\\\"width:130px;height:20px;color:#f9f500\\\">\".concat(params.data[0], \"\\u5927\\u533A<div>\\n                      <div style=\\\"width:130px;height:10px;\\\">\\n                          <div style=\\\"width:120px;display:flex;justify-content:space-between;\\\">\\n                              <span style=\\\"color:#fff\\\">\\u8865\\u8D27\\u53EF\\u5F97\\u7387</span>\\n                              <span style=\\\"color:#0ae9ff\\\">\").concat(params.data[1], \"%</span>\\n                          </div>\\n                      <div>\\n                      <div style=\\\"width:100px;height:10px;\\\">\\n                          <div style=\\\"width:120px;display:flex;justify-content:space-between;\\\">\\n                              <span style=\\\"color:#fff\\\">\\u65E5\\u73AF\\u6BD4</span>\\n                              <span style=\\\"color:#0ae9ff\\\">\").concat(params.data[2], \"%</span>\\n                          </div>\\n                      <div>\\n                  </div>\");\n            return html;\n          }"
    },
    "visualMap": {
      "min": 0,
      "max": 100,
      "left": 26,
      "bottom": 40,
      "showLabel": true,
      "text": [
        "高",
        "低"
      ],
      "inRange": {
        "color": [
          "rgba(19,198,249,0)",
          "rgba(19,198,249,0)"
        ]
      },
      "show": false
    },
    "geo": {
      "map": "china",
      "roam": 1,
      "scaleLimit": {
        "min": 1,
        "max": 2
      },
      "zoom": 1,
      "label": {
        "show": false,
        "color": "#fff",
        "fontSize": 10
      },
      "emphasis": {
        "label": {
          "show": false,
          "color": "#fff",
          "fontSize": 10
        }
      },
      "itemStyle": {
        "normal": {
          "borderColor": "rgba(19,198,249,0.45)",
          "borderWidth": 1,
          "color": "rgba(0,0,0,0)",
          "areaColor": {
            "type": "radial",
            "x": 0.5,
            "y": 0.5,
            "r": 0.8,
            "colorStops": [
              {
                "offset": 0,
                "color": "rgba(0,0,0,0)"
              },
              {
                "offset": 1,
                "color": "rgba(19,198,249,0.15)"
              }
            ],
            "global": false
          },
          "shadowColor": "rgba(19,198,249,1)",
          "shadowOffsetX": 0,
          "shadowOffsetY": 0,
          "shadowBlur": 10
        },
        "emphasis": {
          // "areaColor": "rgba(19,198,249,0)",
          // "shadowOffsetX": 0,
          // "shadowOffsetY": 0,
          // "borderWidth": 0
        }
      },
      "regions": [
        {
          "name": "南海诸岛",
          "itemStyle": {
            "normal": {
              "opacity": 0
            }
          },
          "label": {
            "show": false
          }
        }
      ]
    },
    "series": [
      {
        "type": "effectScatter",
        "coordinateSystem": "geo",
        "showEffectOn": "render",
        "zlevel": 1,
        "symbol": "path://M409,271.9c0,40.2-69.6,73.8-155.5,73.8S98,313.1,98,272.9s69.6-72.8,155.5-72.8S409,231.7,409,271.9z M362,274.3c0,28.2-48.8,51.7-109,51.7s-109-22.8-109-51s48.8-51,109-51S362,246.1,362,274.3z M252.1,309.5c-37.9,0-67.6-14.7-67.6-33.5s29.7-33.5,67.6-33.5c17.7,0,34.3,3.2,46.8,8.9c13.4,6.2,20.8,14.8,20.8,24.2C319.7,294.6,290,309.5,252.1,309.5z M252.1,249.5c-32.9,0-60.6,12.1-60.6,26.5s27.8,26.5,60.6,26.5c16.6,0,32.1-3.1,43.8-8.7c10.7-5.1,16.8-11.7,16.8-18.3C312.7,261.4,285,249.5,252.1,249.5z M250.5,223h-2v-8h2V223z M258.5,215h-2v-8h2V215z M250.5,207h-2v-8h2V207z M258.5,199h-2v-8h2V199z M250.5,191h-2v-8h2V191z M258.5,183h-2v-8h2V183z M250.5,175h-2v-8h2V175z M258.5,167h-2v-8h2V167z M250.5,159h-2v-8h2V159z M258.5,151h-2v-8h2V151z M250.5,143h-2v-8h2V143z M258.5,135h-2v-8h2V135z M250.5,127h-2v-8h2V127z M258.5,119h-2v-8h2V119z M250.5,111h-2v-8h2V111z M258.5,103h-2v-8h2V103z M250.5,95h-2v-8h2V95z M258.5,87h-2v-8h2V87z M250.5,79h-2v-8h2V79z M258.5,71h-2v-8h2V71z M250.5,63h-2v-8h2V63z M258.5,55h-2v-8h2V55z M250.5,47h-2v-5h5v2h-3V47z",
        "rippleEffect": {
          "period": 15,
          "scale": 4,
          "brushType": "fill"
        },
        "hoverAnimation": true,
        "label": {
          "normal": {
            "formatter": "{b}",
            "position": "bottom",
            "offset": [
              0,
              10
            ],
            "color": "#fff",
            "show": true
          }
        },
        "itemStyle": {
          "normal": {
            "color": "#1DE9B6",
            "shadowBlur": 10,
            "shadowColor": "#333"
          }
        },
        "symbolSize": 8,
        "data": [
          {
            "name": "北京",
            "value": [
              116.405285,
              39.904989
            ],
            "itemStyle": {
              "color": "rgba(19,198,249,1)"
            },
            "visualMap": false
          },
          {
            "name": "天津",
            "value": [
              117.190182,
              39.12559
            ],
            "visualMap": false
          },
          {
            "name": "山西",
            "value": [
              112.549248,
              37.857014
            ],
            "visualMap": false
          },
          {
            "name": "内蒙古",
            "value": [
              111.670801,
              40.818311
            ],
            "visualMap": false
          },
          {
            "name": "辽宁",
            "value": [
              123.429096,
              41.796767
            ],
            "visualMap": false
          }
        ]
      },
    ]
  }


option && myChart.setOption(option);})();
