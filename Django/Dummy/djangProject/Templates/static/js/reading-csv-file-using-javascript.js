function buildTable(results){

	var markup = "<table class='table'>";
	var data = results.data;
	var nn;
	console.log(data.length);
	var arr1 = [];
	for(i=0;i<data.length;i++){
		markup+= "<tr>";
		var row = data[i];

		var cells = row.join(",").split(",");


		for(j=1;j<cells.length;j++){
			markup+= "<td>";
			markup+= cells[j];
			nn= parseFloat(cells[j])
			 arr1.push(nn);
			markup+= "</th>";
		}
		markup+= "</tr>";
	}
	markup+= "</table>";



	//var series = [];
	var arr=[]
	for(i=1;i<arr1.length;i++){

			var point ={}
			point["num"]=i;
			point["dataaa"]=arr1[i];


		arr.push(point);
	}

console.log(arr);
	//$("#app").html(arr1);
//brush_data(series);
//arrayToCSV (series);

return arr1;
}




//brushhhhhhhhhhh





function brush_data(data1){
	var data = data1;
  var max_size = parseInt(data.length * 10/100);
	console.log(max_size);
	var optionsline2 = {
		chart: {
			id: 'chart2',
			type: 'line',
			height: 230,
			toolbar: {
				autoSelected: 'pan',
				show: false
			}
		},
		colors: ['#546E7A'],
		stroke: {
			width: 3
		},
		dataLabels: {
			enabled: false
		},
		fill: {
			opacity: 1,
		},
		markers: {
			size: 0
		},
		series: [{
			data: data
		}]

	}

	var chartline2 = new ApexCharts(
		document.querySelector("#chart-line2"),
		optionsline2
	);

	chartline2.render();

	var options = {
		chart: {
			id: 'chart1',
			height: 130,
			type: 'area',
			brush:{
				target: 'chart2',
				enabled: true
			},
			selection: {
				enabled: true,
				xaxis: {
					min: 1,
					max: max_size
				}
			},
		},
		colors: ['#008FFB'],
		series: [{
			data: data
		}],
		fill: {
			type: 'gradient',
			gradient: {
				opacityFrom: 0.91,
				opacityTo: 0.1,
			}
		},

		yaxis: {
			tickAmount: 2
		}
	}

	var chart = new ApexCharts(
		document.querySelector("#chart-line"),
		options
	);

	chart.render();
	//console.log(chart);


}


function Analayse() {

  alert("Clicked");
}




function arrayToCSV (twoDiArray) {
    var csvRows = [];
    for (var i = 0; i < twoDiArray.length; ++i) {
        for (var j = 0; j < twoDiArray[i].length; ++j) {
            twoDiArray[i][j] = '\"' + twoDiArray[i][j] + '\"';
        }
        csvRows.push(twoDiArray[i].join(','));
    }

    var csvString = csvRows.join('\r\n');
    var a         = document.createElement('a');
    a.href        = 'data:attachment/csv,' + csvString;
    //a.target      = '_blank';
		a.target      = 'C:/Megatron/Masters/Web/test2/files';
    a.download    = 'myFile.csv';

    document.body.appendChild(a);
    a.click();
    // Optional: Remove <a> from <body> after done
}





$(document).ready(function(){
		$('#submit').on("click",function(e){

			e.preventDefault();
			if (!$('#files')[0].files.length){
				alert("Please choose at least one file to read the data.");
			}

			$('#files').parse({
				config: {
					delimiter: "auto",
					complete: buildTable,
				},
				before: function(file, inputElem)
				{
					//console.log("Parsing file...", file);
				},
				error: function(err, file)
				{
					console.log("ERROR:", err, file);
				},
				complete: function()
				{
					//console.log("Done with all files");
				}
			});
		});
});
