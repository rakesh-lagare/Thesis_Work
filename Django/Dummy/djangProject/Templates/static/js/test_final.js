function buildTable(results) {
		var filee= document.getElementById('files');
		console.log(filee);
    var data = results.data;
    var row_data;
    var temp_arr = [];
    for (i = 0; i < data.length; i++) {
        var row = data[i];
        var cells = row.join(",").split(",");
        for (j = 1; j < cells.length; j++) {
            row_data = parseFloat(cells[j])
            temp_arr.push(row_data);
        }
    }

    var series = [];
    for (i = 1; i < temp_arr.length; i++) {
        var point = {}
        point["Num"] = i;
        point["Col_Data"] = temp_arr[i];
        series.push(point);
    }


    brush_graph(series);
    return series;
}




$(document).ready(function() {
    $('#submit').on("click", function(e) {
        e.preventDefault();
        if (!$('#files')[0].files.length) {
            alert("Please choose at least one file to read the data.");
        }

        $('#files').parse({
            config: {
                delimiter: "auto",
                complete: buildTable,
            },
            before: function(file, inputElem) {
                //console.log("Parsing file...", file);
            },
            error: function(err, file) {
                console.log("ERROR:", err, file);
            },
            complete: function() {
                //console.log("Done with all files");
            }
        });
    });
});




//////////////////////////////   Graph Brush ///////////////////////////////////////////////////


function brush_graph(series) {
    var selected_segments;
    console.log(series.length);
    var svg = d3.select("svg"),
        margin = {
            top: 20,
            right: 20,
            bottom: 110,
            left: 40
        },
        margin2 = {
            top: 430,
            right: 20,
            bottom: 30,
            left: 40
        },
        width = +svg.attr("width") - margin.left - margin.right,
        height = +svg.attr("height") - margin.top - margin.bottom,
        height2 = +svg.attr("height") - margin2.top - margin2.bottom;

    var parseDate = d3.timeParse("%m/%d/%Y %H:%M");

    var x = d3.scaleLinear().range([0, width]),
        x2 = d3.scaleLinear().range([0, width]),
        y = d3.scaleLinear().range([height, 0]),
        y2 = d3.scaleLinear().range([height2, 0]);

    var xAxis = d3.axisBottom(x),
        xAxis2 = d3.axisBottom(x2),
        yAxis = d3.axisLeft(y);

    var brush = d3.brushX()
        .extent([
            [0, 0],
            [width, height2]
        ])
        .on("brush end", brushed);

    var zoom = d3.zoom()
        .scaleExtent([1, Infinity])
        .translateExtent([
            [0, 0],
            [width, height]
        ])
        .extent([
            [0, 0],
            [width, height]
        ])
        .on("zoom", zoomed);

    var line = d3.line()
        .x(function(d) {
            return x(d.Num);
        })
        .y(function(d) {
            return y(d.Col_Data);
        });

    var line2 = d3.line()
        .x(function(d) {
            return x2(d.Num);
        })
        .y(function(d) {
            return y2(d.Col_Data);
        });

    var clip = svg.append("defs").append("svg:clipPath")
        .attr("id", "clip")
        .append("svg:rect")
        .attr("width", width)
        .attr("height", height)
        .attr("x", 0)
        .attr("y", 0);


    var Line_chart = svg.append("g")
        .attr("class", "focus")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
        .attr("clip-path", "url(#clip)");


    var focus = svg.append("g")
        .attr("class", "focus")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    var context = svg.append("g")
        .attr("class", "context")
        .attr("transform", "translate(" + margin2.left + "," + margin2.top + ")");




    var data = series;


    x.domain(d3.extent(data, function(d) {
        return d.Num;
    }));
    y.domain([0, d3.max(data, function(d) {
        return d.Col_Data;
    })]);
    x2.domain(x.domain());
    y2.domain(y.domain());


    focus.append("g")
        .attr("class", "axis axis--x")
        .attr("transform", "translate(0," + height + ")")
        .call(xAxis);

    focus.append("g")
        .attr("class", "axis axis--y")
        .call(yAxis);

    Line_chart.append("path")
        .datum(data)
        .attr("class", "line")
        .attr("d", line);

    context.append("path")
        .datum(data)
        .attr("class", "line")
        .attr("d", line2);


    context.append("g")
        .attr("class", "axis axis--x")
        .attr("transform", "translate(0," + height2 + ")")
        .call(xAxis2);

    context.append("g")
        .attr("class", "brush")
        .call(brush)
        .call(brush.move, x.range());

    svg.append("rect")
        .attr("class", "zoom")
        .attr("width", width)
        .attr("height", height)
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
        .call(zoom);



    function brushed() {
        if (d3.event.sourceEvent && d3.event.sourceEvent.type === "zoom") return; // ignore brush-by-zoom
        var s = d3.event.selection || x2.range();
        selected_segments = s;

        x.domain(s.map(x2.invert, x2));
        Line_chart.select(".line").attr("d", line);
        focus.select(".axis--x").call(xAxis);
        svg.select(".zoom").call(zoom.transform, d3.zoomIdentity
            .scale(width / (s[1] - s[0]))
            .translate(-s[0], 0));
    }

    function zoomed() {
        if (d3.event.sourceEvent && d3.event.sourceEvent.type === "brush") return; // ignore zoom-by-brush
        var t = d3.event.transform;

        x.domain(t.rescaleX(x2).domain());
        Line_chart.select(".line").attr("d", line);
        focus.select(".axis--x").call(xAxis);
        context.select(".brush").call(brush.move, x.range().map(t.invertX, t));

    }

    function type(d) {

        d.Num = +d.Num;
        d.Col_Data = +d.Col_Data;
        return d;
    }


    function Analayse() {

        var f = selected_segments[0] * series.length / 900
        var s = selected_segments[1] * series.length / 900

        var first = parseInt(f);
        var sec = parseInt(s);
        console.log("first", selected_segments[0])
        console.log("Second", selected_segments[1])
        console.log("first", first)
        console.log("Second", sec)
        //console.log(selected_segments);
        return (first, sec)

    }

    var start_point, end_point = Analayse();
    console.log(start_point, end_point);
    send_data(start_point, end_point);

}



function send_data(start_point, end_point) {
    console.log("start_point and  end_point");
    console.log(start_point, end_point);
}
