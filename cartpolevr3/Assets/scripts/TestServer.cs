using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Net.Sockets;
using System.Net;
using System.Threading;
using System.Runtime.InteropServices;
using System.IO;
using System.Text;

public struct SendPacket
{
    public float data1; //x위치
    public float data2; //x속도


    public float data3; //z각도
    public float data4; //z각속도

    public bool isDone; //종료확인
    public float hight;
}

public struct RecPacket
{
    public float data; //x 방향힘
}

class ThreadParam
{
    public TcpClient param1;
    public int param2;
    public ThreadParam(TcpClient num1, int num2)
    {
        this.param1 = num1;
        this.param2 = num2;
    }
} 

class Control_socket
{
    public int index_process = 0;
    public int nun_of_process;
    public TcpClient[] each_socket = new TcpClient[100];
    public Thread[] process = new Thread[100];
    public bool[] Isconnected = new bool[100];
    public NetworkStream[] process_Stream = new NetworkStream[100];
    public RecPacket[] StoreRecData = new RecPacket[100];
    public bool[] IsStore = new bool[100];
    public bool[] IsRecStore = new bool[100];

    public Control_socket( int nun_of_process)
    {
        this.nun_of_process = nun_of_process;
        for(int i = 0; i < nun_of_process; i++)
        {
            Isconnected[i] = false;
            IsStore[i] = false;
            IsRecStore[i] = false;
        }

    }


}


public class TestServer : MonoBehaviour
{
    // Start is called before the first frame update
    public static TestServer instance;
    TcpListener Server = null;
    private int whattime = 0;
    //bool Socket_Threading_Flag = false;

    //SendPacket SendData;
    RecPacket RecData;
    bool isReced;
    Thread Socket_Thread;

    public Testcontroller[] controllers = new Testcontroller[5];
    private int num_process = 5;
    private int index_process = 0;

    Control_socket Control_Socket;



    private void Awake()
    {
        Control_Socket = new Control_socket(num_process);

        instance = this;
        Socket_Thread = new Thread(makeServer);
        //Socket_Threading_Flag = true;
        Socket_Thread.Start();


    }

    private void FixedUpdate()
    {

        int index = 0;
        while(index < num_process)
        {
            if (Control_Socket.IsRecStore[index])
            {
                Debug.Log("data 받음");//해당 인덱스를 프로세서로 가진 컨트롤러만 작동

                StartCoroutine(controllers[index].Action(Control_Socket.StoreRecData[index].data));
                Control_Socket.IsRecStore[index] = false;

            }


            index++;
        }
    }
    public static T ByteToStruct<T>(byte[] buffer) where T : struct
    {
        int size = Marshal.SizeOf(typeof(T));
        //Debug.Log(System.String.Format("size is {0} buffer.Length is {1}", size, buffer.Length));
        if (size > buffer.Length)
        {
            throw new Exception();
        }

        IntPtr ptr = Marshal.AllocHGlobal(size);
        Marshal.Copy(buffer, 0, ptr, size);
        T obj = (T)Marshal.PtrToStructure(ptr, typeof(T));
        Marshal.FreeHGlobal(ptr);
        return obj;
    }

    public static byte[] StructToByte(object obj)
    {
        int datasize = Marshal.SizeOf(obj);
        IntPtr buff = Marshal.AllocHGlobal(datasize);
        Marshal.StructureToPtr(obj, buff, false);
        byte[] data = new byte[datasize];
        Marshal.Copy(buff, data, 0, datasize);
        Marshal.FreeHGlobal(buff);
        return data;
    }

    private void makeServer()
    {
        Int32 Port = 1111;
        IPAddress Addr = IPAddress.Any;
        Server = new TcpListener(Addr, Port);
        Server.Start();
        while (index_process < num_process)
        {
            Control_Socket.each_socket[index_process] = Server.AcceptTcpClient();
            Control_Socket.process[index_process] = new Thread(docker);
            ThreadParam tp = new ThreadParam(Control_Socket.each_socket[index_process], index_process);
            Control_Socket.process[index_process].Start(tp);
            index_process = index_process + 1;
        }
    }


    private void docker(object obj)
    {
        ThreadParam tempParam = obj as ThreadParam;

        TcpClient Client = tempParam.param1;
        int index = tempParam.param2;
        //Debug.Log("소켓 대기중....");
        //Debug.Log("소켓 연결되었습니다.");
        Control_Socket.process_Stream[index] = Client.GetStream();
        Control_Socket.Isconnected[index] = true;
        byte[] Buffer = new byte[1024];

        int length = 0;
        while (true)
        {
            //Debug.Log(".............................");
            try
            {
                //Debug.Log("데이터 받는중 ");

                while(true)
                {
       

                    length = Control_Socket.process_Stream[index].Read(Buffer, 0, 1023);
                    //Debug.Log(length);

                    RecData = ByteToStruct<RecPacket>(Buffer);

                    Control_Socket.StoreRecData[index] = RecData;
                    Control_Socket.IsRecStore[index] = true;
                    Debug.Log("데이터 받음 process "+ index);

                }

            }
            catch (Exception e)
            {
                Debug.Log(e.Message);
                //ocket_Threading_Flag = false;
                Client.Close();
                Server.Stop();
                continue;
            }
        }
    }
    public void SendMessage(SendPacket sendData, int process_index)
    {

        //Debug.Log(Marshal.SizeOf(sendData) + " : " + sendData.data1 + " " + sendData.data2 + " " + sendData.data3 + " " + sendData.data4 + " " + sendData.data5 + " " + sendData.data6 + " " + sendData.data7);
        byte[] packetArray = StructToByte(sendData);
        //if(sendStm != null)
        //Debug.Log(whattime);
        Control_Socket.process_Stream[process_index].Write(packetArray, 0, packetArray.Length);
        // Debug.Log(System.String.Format("send massage process : {0} {1} {2} {3}", process_index, sendData.data1 , sendData.data2, sendData.data3));
    }

    public bool CheckConnected(int process_index)
    {
        return Control_Socket.Isconnected[process_index];
    }
}
