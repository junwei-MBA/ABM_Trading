const HistogramModule = function(bins, canvas_width, canvas_height,type) {
  // Create the canvas object:
    const canvas = document.createElement("canvas");
    Object.assign(canvas, {
      width: canvas_width,
      height: canvas_height,
      style: "border:1px dotted",
    });

  // Append it to #elements:
  const elements = document.getElementById("elements");
  // elements.appendChild(canvas);
  //自定义开始
  const div = document.createElement("div");
  Object.assign(div, {
    className:'customDiv',
  });
  div.appendChild(canvas);
  elements.appendChild(div);
  //自定义结束


  // Create the context and the drawing controller:
  const context = canvas.getContext("2d");

  // Prep the chart properties and series:
  const datasets = [{
    label: type,
    fillColor: "rgba(151,187,205,0.5)",
    strokeColor: "rgba(151,187,205,0.8)",
    highlightFill: "rgba(151,187,205,0.75)",
    highlightStroke: "rgba(151,187,205,1)",
    data: []
  }];

  // Add a zero value for each bin
  for (var i in bins)
    datasets[0].data.push(0);

  const data = {
    labels: bins,
    datasets: datasets
  };

  const options = {
    scaleBeginsAtZero: true
  };

  // Create the chart object
  let chart = new Chart(context, {type: 'bar', data: data, options: options});

  // Now what?
  this.render = function(data) {
    datasets[0].data = data;
    chart.update();
  };

  this.reset = function() {
    chart.destroy();
    chart = new Chart(context, {type: 'bar', data: data, options: options});
  };
};