window.onload = function () {

    var chart = new CanvasJS.Chart("chartContainer", {
        theme: "dark2", // "light2", "dark1", "dark2"
        animationEnabled: false, // change to true		
        title: {
            text: "apple"
        },
        data: [
            {
                // Change type to "bar", "area", "spline", "pie",etc.
                type: "column",
                dataPoints: [
                    { label: "1차", y: 0.76 },
                    { label: "2차", y: 0.82 },
                    { label: "3차", y: 0.89 },
                    { label: "4차", y: 0.91 },
                    { label: "5차", y: 0.95 }
                ]
            }
        ]
    });
    chart.render();
}