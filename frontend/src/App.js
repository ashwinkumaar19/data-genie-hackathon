import './App.css';
import React from 'react';
import Plot from 'react-plotly.js';
import { useState, useEffect } from 'react';
import axios from 'axios';
import Papa from 'papaparse';


function App() {

  const [fromDate, setFromData] = useState('');
  const [toDate, setToData] = useState('');
  const [period, setPeriod] = useState('0');
  const [mape, setMape] = useState(0.0);

  const [graphData, setGraphData] = useState([]);


  const [csvData, setCsvData] = useState([]);

  const handleOnDrop = (event) => {
    const file = event.target.files[0];
    const tempList = []
    Papa.parse(file, {
      header: true,
      complete: (results) => {
        results.data.forEach(element => {
          const temp = {
            "point_timestamp": element.point_timestamp,
            "point_value": element.point_value
          }
          tempList.push(temp)
        });
        console.log(tempList)
        setCsvData(tempList);
      },
    });
  };

  const onSubmit = async () => {
    const resp = await axios.post(`http://localhost:8000/predict?date_from=${fromDate}&date_to=${toDate}&period=${period}`, {
      data: csvData,
      header: {
        'Access-Control-Allow-Origin': '*',
      }
    });

    if(!resp.error){
      const tempData = resp.data;
      console.log(" ---  mape --", tempData?.mape)
      setMape(tempData?.mape);
      const tempResults = tempData?.result;
      const tempX = [];
      const tempY1 = [];
      const tempY2 = [];
      tempResults.forEach((item)=>{
          tempX.push(item.point_timestamp);
          tempY1.push(item.y);
          tempY2.push(item.yhat);
      })

      setGraphData([
        {
          x: tempX,
          y: tempY1,
          type: 'scatter',
          mode: 'lines+markers',
          marker: {color: 'red'},
          name: 'actual'
        },
        {
          x: tempX,
          y: tempY2,
          type: 'scatter',
          mode: 'lines+markers',
          marker: {color: 'green'},
          name: 'predicted'
        },
      ])
    }
  }

  const layout = {
    title: 'Line Graph',
    xaxis: {
      title: 'X Axis',
    },
    yaxis: {
      title: 'Y Axis',
    },
  };


  return (
    <div className="App" style={{padding: '24px'}}>

    <div style={{flex: 'row', display: 'flex', justifyContent:'center', alignItems: 'center', margin: '10px 0 0 0'}}>
      <div>From Data:</div>
      <input type={'date'} style={{padding: '6px', margin: '0px 10px' }} onChange={(e)=>{setFromData(e.target.value)}}/>
    </div>

    <div style={{flex: 'row', display: 'flex', justifyContent:'center', alignItems: 'center', margin: '10px 0 0 0'}}>
      <div>To Data:</div>
      <input type={'date'} style={{padding: '6px', margin: '0px 10px' }} onChange={(e)=>{setToData(e.target.value)}}/>
    </div>

    <div style={{flex: 'row', display: 'flex', justifyContent:'center', alignItems: 'center', margin: '10px 0 0 0'}}>
      <div>Period:</div>
      <input type={'number'} style={{padding: '6px', margin: '0px 10px' }} onChange={(e)=>{setPeriod(e.target.value)}}/>
    </div>

    <div style={{flex: 'row', display: 'flex', justifyContent:'center', alignItems: 'center', margin: '10px 0 0 0'}}>
      <div>Upload CSV file:</div>
      <input type={'file'} style={{padding: '6px', margin: '0px 10px' }} onChange={handleOnDrop}/>
    </div>

    <div>
      <button onClick={onSubmit}>Submit</button>
    </div>

    {graphData.length> 0 && 
    <div style={{lex: 'row', display: 'flex', justifyContent:'center', alignItems: 'center'}}>
      <Plot
        data={graphData}
        layout={layout}
        style={{width: '100%', height: '100%'}}
      />
    </div>
    }
    {graphData.length> 0 && 

      <div style={{flex: 'row', display: 'flex', justifyContent:'center', alignItems: 'center', margin: '10px 0 0 0'}}>
        <div>MAPE :</div>
        <div>{mape}</div>
      </div>
    }

    
    </div>
  );
}

export default App;