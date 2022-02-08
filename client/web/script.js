// Connect to the WebSocket Server
socket = io.connect('http://' + window.location.hostname + ':9090');

// Receive the input image
socket.on('input', function (data) {
    $('#input').attr("src", "data:image/png;base64," + data.data);
});

// Receive the adv image
socket.on('adv', function (data) {
    $('#adv').attr("src", "data:image/png;base64," + data.data);
});

socket.on('connect', function () {
    console.log('Client has connected to the server!');
    stop_attack();
});

socket.on('disconnect', function () {
    console.log('The client has disconnected!');
    $("#customSwitchActivate").prop("checked", false);
});

function clear_patch() {
    socket.emit('clear_patch', 1);
}

// Receive the original image
socket.on('update', function (data) {
    $('#origin').attr("src", "data:image/png;base64," + data.data);
});

function save_patch() {
    socket.emit('save_patch');
}

function start_attack() {
    socket.emit('fix_patch', 0);
}

function pause_attack() {
    socket.emit('fix_patch', 1);
}

function stop_attack() {
    socket.emit('fix_patch', 1);
    socket.emit('clear_patch', 1);
}

$(document).ready(function () {
    
});
