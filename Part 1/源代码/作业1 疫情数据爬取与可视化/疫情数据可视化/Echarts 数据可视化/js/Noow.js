(function () {
    // 1. 实例化对象
    var myChart = echarts.init(document.querySelector(".line2 .chart"));
    // 2. 指定配置选项 和数据
    option = {
        tooltip: {
            trigger: "axis",
            axisPointer: {
                lineStyle: {
                    color: "#dddc6b"
                }
            }
        },
        legend: {
            top: "0%",
            textStyle: {
                color: "rgba(255,255,255,.5)",
                fontSize: "12"
            }
        },
        grid: {
            left: "10",
            top: "30",
            right: "10",
            bottom: "10",
            containLabel: true
        },

        xAxis: [
            {
                type: "category",
                boundaryGap: false,
                axisLabel: {
                    textStyle: {
                        color: "rgba(255,255,255,.6)",
                        fontSize: 12
                    }
                },
                axisLine: {
                    lineStyle: {
                        color: "rgba(255,255,255,.2)"
                    }
                },

                data: [
                    "01",
                    "02",
                    "03",
                    "04",
                    "05",
                    "06",
                    "07",
                    "08",
                    "09",
                    "11",
                    "12",
                    "13",
                    "14",
                    "15",
                    "16",
                    "17",
                    "18",
                    "19",
                    "20",
                    "21",
                    "22",
                    "23",
                    "24",
                    "25",
                    "26",
                    "27",
                    "28",
                    "29",
                    "30"
                ]
            },
            {
                axisPointer: { show: false },
                axisLine: { show: false },
                position: "bottom",
                offset: 20
            }
        ],

        yAxis: [
            {
                type: "value",
                axisTick: { show: false },
                axisLine: {
                    lineStyle: {
                        color: "rgba(255,255,255,.1)"
                    }
                },
                axisLabel: {
                    textStyle: {
                        color: "rgba(255,255,255,.6)",
                        fontSize: 12
                    }
                },

                splitLine: {
                    lineStyle: {
                        color: "rgba(255,255,255,.1)"
                    }
                }
            }
        ],
        series: [
            {
                name: "播放量",
                type: "line",
                smooth: true,
                symbol: "circle",
                symbolSize: 5,
                showSymbol: false,
                lineStyle: {
                    normal: {
                        color: "#0184d5",
                        width: 2
                    }
                },
                areaStyle: {
                    normal: {
                        color: new echarts.graphic.LinearGradient(
                            0,
                            0,
                            0,
                            1,
                            [
                                {
                                    offset: 0,
                                    color: "rgba(1, 132, 213, 0.4)"
                                },
                                {
                                    offset: 0.8,
                                    color: "rgba(1, 132, 213, 0.1)"
                                }
                            ],
                            false
                        ),
                        shadowColor: "rgba(0, 0, 0, 0.1)"
                    }
                },
                itemStyle: {
                    normal: {
                        color: "#0184d5",
                        borderColor: "rgba(221, 220, 107, .1)",
                        borderWidth: 12
                    }
                },
                data: [
                    30,
                    40,
                    30,
                    40,
                    30,
                    40,
                    30,
                    60,
                    20,
                    40,
                    20,
                    40,
                    30,
                    40,
                    30,
                    40,
                    30,
                    40,
                    30,
                    60,
                    20,
                    40,
                    20,
                    40,
                    30,
                    60,
                    20,
                    40,
                    20,
                    40
                ]
            },
            {
                name: "转发量",
                type: "line",
                smooth: true,
                symbol: "circle",
                symbolSize: 5,
                showSymbol: false,
                lineStyle: {
                    normal: {
                        color: "#00d887",
                        width: 2
                    }
                },
                areaStyle: {
                    normal: {
                        color: new echarts.graphic.LinearGradient(
                            0,
                            0,
                            0,
                            1,
                            [
                                {
                                    offset: 0,
                                    color: "rgba(0, 216, 135, 0.4)"
                                },
                                {
                                    offset: 0.8,
                                    color: "rgba(0, 216, 135, 0.1)"
                                }
                            ],
                            false
                        ),
                        shadowColor: "rgba(0, 0, 0, 0.1)"
                    }
                },
                itemStyle: {
                    normal: {
                        color: "#00d887",
                        borderColor: "rgba(221, 220, 107, .1)",
                        borderWidth: 12
                    }
                },
                data: [
                    50,
                    30,
                    50,
                    60,
                    10,
                    50,
                    30,
                    50,
                    60,
                    40,
                    60,
                    40,
                    80,
                    30,
                    50,
                    60,
                    10,
                    50,
                    30,
                    70,
                    20,
                    50,
                    10,
                    40,
                    50,
                    30,
                    70,
                    20,
                    50,
                    10,
                    40
                ]
            }
        ]
    };

    // 使用刚指定的配置项和数据显示图表。
    myChart.setOption(option);
    // 让我们的图表适配屏幕宽度
    window.addEventListener("resize", function () {
        //   让我们图表调用resize方法
        myChart.resize();
    });
})();