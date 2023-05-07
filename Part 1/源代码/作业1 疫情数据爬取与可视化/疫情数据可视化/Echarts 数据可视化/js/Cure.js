(function() {
    // 1. 实例化对象
    var myChart = echarts.init(document.querySelector(".line .chart"));
    // 2. 指定配置选项 和数据
    option = {
        tooltip: {
            trigger: 'axis'
        },
        legend: {
            data: ['美国', '中国', '日本', '俄罗斯', '德国'],
            textStyle: {
                color: "#ffffff",
                fontSize: "12"
            },
            color:"#ffffff"
        },
        grid: {
            left: '3%',
            right: '4%',
            bottom: '3%',
            containLabel: true
        },
        toolbox: {

        },
        label: {
            color: '#ffffff'
        },
        xAxis: {
            type: 'category',
            boundaryGap: true,
            data: ['3.19', '3.20', '3.21', '3.22', '3.23', '3.24', '3.25'],
            axisLabel: {
                color: "#5677fc",
                fontSize: "11"
            },
            axisLine: {
                color:"#ffffff"
            }
        },
        yAxis: {
            type: 'value',
            axisLabel: {
                color: "#5677fc",
                fontSize: "11"
            },
            axisLine: {
                color:"#ffffff"
            }
        },
        series: [
            {
                name: '美国',
                type: 'line',
                stack: '总量',
                data: [120, 132, 101, 134, 90, 230, 210]
            },
            {
                name: '中国',
                type: 'line',
                stack: '总量',
                data: [220, 182, 191, 234, 290, 330, 310]
            },
            {
                name: '日本',
                type: 'line',
                stack: '总量',
                data: [150, 232, 201, 154, 190, 330, 410]
            },
            {
                name: '俄罗斯',
                type: 'line',
                stack: '总量',
                data: [320, 332, 301, 334, 390, 330, 320]
            },
            {
                name: '德国',
                type: 'line',
                stack: '总量',
                data: [820, 932, 901, 934, 1290, 1330, 1320]
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