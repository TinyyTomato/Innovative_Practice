const mysql = require('mysql');

//创建连接对象
const connection = mysql.createConnection({
    host:'localhost',
    user:'root',
    password:'luomy1010',
    database:'xinguan'
})

//开始连接
connection.connect();


date=['2020.2.22', '2020.2.23', '2020.2.24', '2020.2.25', '2020.2.26', '2020.2.27', '2020.2.28', '2020.2.29', '2020.3.1', '2020.3.2', '2020.3.3', '2020.3.4', '2020.3.5', '2020.3.6', '2020.3.7', '2020.3.8', '2020.3.9', '2020.3.10', '2020.3.11', '2020.3.12', '2020.3.13', '2020.3.14', '2020.3.15', '2020.3.16', '2020.3.17', '2020.3.18', '2020.3.19', '2020.3.20', '2020.3.21', '2020.3.22', '2020.3.23', '2020.3.24', '2020.3.25', '2020.3.26', '2020.3.27', '2020.3.28', '2020.3.29', '2020.3.30', '2020.3.31', '2020.4.1', '2020.4.2', '2020.4.3', '2020.4.4', '2020.4.5', '2020.4.6', '2020.4.7', '2020.4.8', '2020.4.9', '2020.4.10', '2020.4.11', '2020.4.12', '2020.4.13', '2020.4.14', '2020.4.15', '2020.4.16', '2020.4.17', '2020.4.18', '2020.4.19', '2020.4.20', '2020.4.21', '2020.4.22', '2020.4.23', '2020.4.24', '2020.4.25', '2020.4.26', '2020.4.27', '2020.4.28', '2020.4.29', '2020.4.30', '2020.5.1', '2020.5.2', '2020.5.3', '2020.5.4', '2020.5.5', '2020.5.6', '2020.5.7', '2020.5.8', '2020.5.9', '2020.5.10', '2020.5.11', '2020.5.12', '2020.5.13', '2020.5.14', '2020.5.15', '2020.5.16', '2020.5.17', '2020.5.18', '2020.5.19', '2020.5.20', '2020.5.21', '2020.5.22', '2020.5.23', '2020.5.24', '2020.5.25', '2020.5.26', '2020.5.27', '2020.5.28', '2020.5.29', '2020.5.30', '2020.5.31', '2020.6.1', '2020.6.2', '2020.6.3', '2020.6.4', '2020.6.5', '2020.6.6', '2020.6.7', '2020.6.8', '2020.6.9', '2020.6.10', '2020.6.11', '2020.6.12', '2020.6.13', '2020.6.14', '2020.6.15', '2020.6.16', '2020.6.17', '2020.6.18', '2020.6.19', '2020.6.20', '2020.6.21', '2020.6.22', '2020.6.23', '2020.6.24', '2020.6.25', '2020.6.26', '2020.6.27', '2020.6.28', '2020.6.29', '2020.6.30', '2020.7.1', '2020.7.2', '2020.7.3', '2020.7.4', '2020.7.5', '2020.7.6', '2020.7.7', '2020.7.8', '2020.7.9', '2020.7.10', '2020.7.11', '2020.7.12', '2020.7.13', '2020.7.14', '2020.7.15', '2020.7.16', '2020.7.17', '2020.7.18', '2020.7.19', '2020.7.20', '2020.7.21', '2020.7.22', '2020.7.23', '2020.7.24', '2020.7.25', '2020.7.26', '2020.7.27', '2020.7.28', '2020.7.29', '2020.7.30', '2020.7.31', '2020.8.1', '2020.8.2', '2020.8.3', '2020.8.4', '2020.8.5', '2020.8.6', '2020.8.7', '2020.8.8', '2020.8.9', '2020.8.10', '2020.8.11', '2020.8.12', '2020.8.13', '2020.8.14', '2020.8.15', '2020.8.16', '2020.8.17', '2020.8.18', '2020.8.19', '2020.8.20', '2020.8.21', '2020.8.22', '2020.8.23', '2020.8.24', '2020.8.25', '2020.8.26', '2020.8.27', '2020.8.28', '2020.8.29', '2020.8.30', '2020.8.31', '2020.9.1', '2020.9.2', '2020.9.3', '2020.9.4', '2020.9.5', '2020.9.6', '2020.9.7', '2020.9.8', '2020.9.9', '2020.9.10', '2020.9.11', '2020.9.12', '2020.9.13', '2020.9.14', '2020.9.15', '2020.9.16', '2020.9.17', '2020.9.18', '2020.9.19', '2020.9.20', '2020.9.21', '2020.9.22', '2020.9.23', '2020.9.24', '2020.9.25', '2020.9.26', '2020.9.27', '2020.9.28', '2020.9.29', '2020.9.30', '2020.10.1', '2020.10.2', '2020.10.3', '2020.10.4', '2020.10.5', '2020.10.6', '2020.10.7', '2020.10.8', '2020.10.9', '2020.10.10', '2020.10.11', '2020.10.12', '2020.10.13', '2020.10.14', '2020.10.15', '2020.10.16', '2020.10.17', '2020.10.18', '2020.10.19', '2020.10.20', '2020.10.21', '2020.10.22', '2020.10.23', '2020.10.24', '2020.10.25', '2020.10.26', '2020.10.27', '2020.10.28', '2020.10.29', '2020.10.30', '2020.10.31', '2020.11.1', '2020.11.2', '2020.11.3', '2020.11.4', '2020.11.5', '2020.11.6', '2020.11.7', '2020.11.8', '2020.11.9', '2020.11.10', '2020.11.11', '2020.11.12', '2020.11.13', '2020.11.14', '2020.11.15', '2020.11.16', '2020.11.17', '2020.11.18', '2020.11.19', '2020.11.20', '2020.11.21', '2020.11.22', '2020.11.23', '2020.11.24', '2020.11.25', '2020.11.26', '2020.11.27', '2020.11.28', '2020.11.29', '2020.11.30', '2020.12.1', '2020.12.2', '2020.12.3', '2020.12.4', '2020.12.5', '2020.12.6', '2020.12.7', '2020.12.8', '2020.12.9', '2020.12.10', '2020.12.11', '2020.12.12', '2020.12.13', '2020.12.14', '2020.12.15', '2020.12.16', '2020.12.17', '2020.12.18', '2020.12.19', '2020.12.20', '2020.12.21', '2020.12.22', '2020.12.23', '2020.12.24', '2020.12.25', '2020.12.26', '2020.12.27', '2020.12.28', '2020.12.29', '2020.12.30', '2020.12.31', '2021.1.1', '2021.1.2', '2021.1.3', '2021.1.4', '2021.1.5', '2021.1.6', '2021.1.7', '2021.1.8', '2021.1.9', '2021.1.10', '2021.1.11', '2021.1.12', '2021.1.13', '2021.1.14', '2021.1.15', '2021.1.16', '2021.1.17', '2021.1.18', '2021.1.19', '2021.1.20', '2021.1.21', '2021.1.22', '2021.1.23', '2021.1.24', '2021.1.25', '2021.1.26', '2021.1.27', '2021.1.28', '2021.1.29', '2021.1.30', '2021.1.31', '2021.2.1', '2021.2.2', '2021.2.3', '2021.2.4', '2021.2.5', '2021.2.6', '2021.2.7', '2021.2.8', '2021.2.9', '2021.2.10', '2021.2.11', '2021.2.12', '2021.2.13', '2021.2.14', '2021.2.15', '2021.2.16', '2021.2.17', '2021.2.18', '2021.2.19', '2021.2.20', '2021.2.21', '2021.2.22', '2021.2.23', '2021.2.24', '2021.2.25', '2021.2.26', '2021.2.27', '2021.2.28', '2021.3.1', '2021.3.2', '2021.3.3', '2021.3.4', '2021.3.5', '2021.3.6', '2021.3.7', '2021.3.8', '2021.3.9', '2021.3.10', '2021.3.11', '2021.3.12', '2021.3.13', '2021.3.14', '2021.3.15', '2021.3.16', '2021.3.17', '2021.3.18', '2021.3.19', '2021.3.20', '2021.3.21', '2021.3.22', '2021.3.23', '2021.3.24', '2021.3.25', '2021.3.26', '2021.3.27', '2021.3.28', '2021.3.29', '2021.3.30']


var options = [];
for (var i=0;i<2;i++){
    str = 'select name,value from xinguan where date="'+date[i]+'"';
    //执行sql语句
    const sql =str;
    var p = connection.query(sql,(err, result)=> {

        if (err) {
            console.log("error", err);
        }else{
            // console.log(JSON.stringify(result));
            options.push(JSON.parse(JSON.stringify(result)));
            // console.log(options);
            return options;
        }

    })

    // option.push(JSON.parse(q));
}
console.log(options);
console.log(p);

//关闭连接
connection.end();