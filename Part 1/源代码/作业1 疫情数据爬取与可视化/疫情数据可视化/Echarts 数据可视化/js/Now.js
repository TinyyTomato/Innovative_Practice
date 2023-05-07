(function() {
    // 1. 实例化对象
    var myChart = echarts.init(document.querySelector(".pie .chart"));
    // 2. 指定配置选项 和数据
    option = {
        // backgroundColor: '#2c343c',

        // title: {
        //     text: 'Customized Pie',
        //     left: 'center',
        //     top: 20,
        //     textStyle: {
        //         color: '#ccc'
        //     }
        // },

        tooltip: {
            trigger: 'item'
        },

        visualMap: {
            show: false,
            min: 80,
            max: 600,
            inRange: {
                colorLightness: [0, 1]
            }
        },
        series: [
            {
                name: '累计人数',
                type: 'pie',
                radius: '55%',
                center: ['50%', '50%'],
                data: [
                    {value: 335, name: '美国'},
                    {value: 310, name: '中国'},
                    {value: 274, name: '日本'},
                    {value: 235, name: '英国'},
                    {value: 400, name: '俄罗斯'}
                ].sort(function (a, b) { return a.value - b.value; }),
                roseType: 'radius',
                label: {
                    color: '#ffffff'
                },
                labelLine: {
                    lineStyle: {
                        color: 'rgba(255, 255, 255, 0.3)'
                    },
                    smooth: 0.2,
                    length: 10,
                    length2: 20
                },
                itemStyle: {
                    color: '#1a237e',
                    shadowBlur: 200,
                    shadowColor: 'rgba(0, 0, 0, 0.5)'
                },

                animationType: 'scale',
                animationEasing: 'elasticOut',
                animationDelay: function (idx) {
                    return Math.random() * 200;
                }
            }
        ]
    };

    option && myChart.setOption(option);

    // 让我们的图表适配屏幕宽度
    window.addEventListener("resize", function() {
        //   让我们图表调用resize方法
        myChart.resize();
    });
})();