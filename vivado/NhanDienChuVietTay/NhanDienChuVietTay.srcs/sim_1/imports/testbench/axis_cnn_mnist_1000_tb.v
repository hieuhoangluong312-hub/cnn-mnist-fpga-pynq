`timescale 1ns / 10ps

module axis_cnn_mnist_1000_tb();
    localparam T = 10; // 100 MHz

    reg aclk;
    reg aresetn;
    wire s_axis_tready;
    reg [7:0] s_axis_tdata;
    reg s_axis_tvalid;
    reg m_axis_tready;
    wire [7:0] m_axis_tdata;
    wire m_axis_tvalid;
    wire m_axis_tlast;

    reg [7:0] s_axis_tdata_reg;
    reg s_axis_tvalid_reg;

    reg [7:0] pixels [0:783999];
    integer i = 0, j = 0;
    reg [9:0] accuracy;

    axis_cnn_mnist dut
    (
        .aclk(aclk),
        .aresetn(aresetn),
        .s_axis_tready(s_axis_tready),
        .s_axis_tdata(s_axis_tdata_reg),
        .s_axis_tvalid(s_axis_tvalid_reg),
        .m_axis_tready(m_axis_tready),
        .m_axis_tdata(m_axis_tdata),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tlast(m_axis_tlast)
    );

    always
    begin
        aclk = 0;
        #(T/2);
        aclk = 1;
        #(T/2);
    end

    always @(posedge aclk)
    begin
        if (!aresetn)
        begin
            s_axis_tdata_reg <= 0;
            s_axis_tvalid_reg <= 0;
        end
        else
        begin
            s_axis_tdata_reg <= s_axis_tdata;
            s_axis_tvalid_reg <= s_axis_tvalid;
        end
    end

    initial
    begin
        $readmemh("input_1000.txt", pixels);

        s_axis_tdata = 0;
        s_axis_tvalid = 0;
        m_axis_tready = 1;

        aresetn = 1'b0;
        #(T*5);
        aresetn = 1'b1;
        #(T*5);
        
        for (j = 0; j < 1000; j=j+1)
        begin
            s_axis_tvalid = 1;
            for (i = 0; i < 784; i=i+1)
            begin
                s_axis_tdata = pixels[j*784+i];
                #T;
            end
            s_axis_tvalid = 0;
            while (s_axis_tready == 1)
            begin
                #T;
            end
            while (s_axis_tready == 0)
            begin
                #T;
            end
        end

        $display("------ Final Accuracy for 1000 Input Image ------");
        $display("Accuracy : %3d%%", accuracy/10);
        $stop;
    end

    always @(posedge aclk)
    begin
        if (!aresetn)
        begin
            accuracy <= 0;
        end
        else if (m_axis_tvalid)
        begin
            if (m_axis_tdata == j%10)
            begin
                $display("Input image %0d: original value = %0d, decision = %0d ==> Success", j, j%10, m_axis_tdata);
                accuracy <= accuracy + 1'b1;
            end
            else
            begin
                $display("Input image %0d: original value = %0d, decision = %0d ==> Fail", j, j%10, m_axis_tdata);
            end
        end
    end

endmodule
