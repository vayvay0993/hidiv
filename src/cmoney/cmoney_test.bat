cd src 
 cd cmoney 
 "CMTrans.exe" "SQL1; select [日期], [股票代號], [股票名稱], [開盤價], [最高價], [最低價], [收盤價] from [日收盤表排行] where ([日期] between '20200101' and '20200130' ) and ([股票代號] in <CM代號,X1> or [股票代號] in <CM代號,X2>  or [股票代號] in <CM代號,1> or [股票代號] in <CM代號,2>) order by [日期], [股票代號] asc  ;,;cmoney_data.txt"