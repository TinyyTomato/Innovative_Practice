(function () {
    // 1. 实例化对象
    var myChart = echarts.init(document.querySelector(".pie2 .chart"));
    // 2. 指定配置选项 和数据
    option = {
        tooltip: {
            trigger: "item",
            formatter: "{a} <br/>{b}: {c} ({d}%)",
            position: function(p) {
                //其中p为当前鼠标的位置
                return [p[0] + 10, p[1] - 10];
            }
        },
        legend: {
            top: "90%",
            itemWidth: 10,
            itemHeight: 10,
            data: ["美国", "中国", "日本", "俄罗斯", "法国"],
            textStyle: {
                color: "rgba(255,255,255,.5)",
                fontSize: "12"
            }
        },
        series: [
            {
                name: "国家",
                type: "pie",
                center: ["50%", "42%"],
                radius: ["40%", "60%"],
                color: [
                    "#065aab",
                    "#066eab",
                    "#0682ab",
                    "#0696ab",
                    "#06a0ab",
                    "#06b4ab",
                    "#06c8ab",
                    "#06dcab",
                    "#06f0ab"
                ],
                label: { show: false },
                labelLine: { show: false },
                data: [
                    { value: 1, name: "美国" },
                    { value: 4, name: "中国" },
                    { value: 2, name: "日本" },
                    { value: 2, name: "俄罗斯" },
                    { value: 1, name: "法国" }
                ]
            }
        ]
    };
    myChart.setOption(option);
    // 让我们的图表适配屏幕宽度
    window.addEventListener("resize", function () {
        //   让我们图表调用resize方法
        myChart.resize();
    });
})();